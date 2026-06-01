#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mediate_local.py -- B4 local-VLM mediation: reconcile multiple OCR reads of a column
into one best transcription, using a served local VLM that SEES the crop + the candidate
reads. Comma/dash-preserving, no-paraphrase prompt. Writes a pseudo-engine the leaderboard
scores like any other.

Modes:
  --mode vlm  : column-level VLM mediation (needs a served mediator on :8090). For each
                eval column, prompt = crop image + the candidate transcriptions; output =
                the reconciled column. -> reads_eval/_vlm/<page_id>__mediate-<mediator>.txt
  --mode ce   : CPU consensus-entropy over the voters' columns (no GPU); picks the
                most-central voter line-set. -> ..._mediate-ce.txt

  python mediate_local.py --mode vlm --mediator Qwen3-VL-8B --voters dots.ocr,PaddleOCR-VL-1.6,tesseract
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, csv, glob, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_h = os.environ.get("CONSENSUS_HARNESS")
if _h:
    sys.path.insert(0, _h)
import bakeoff  # clean(), ref_for, eval_ids  # noqa: E402

PROJ = _HERE.parent.parent
VLM_RAW = PROJ / "Outputs" / "reads_eval" / "_vlm"
READS_EVAL = PROJ / "Outputs" / "reads_eval"
ENDPOINT = "http://127.0.0.1:8090"

MEDIATE_PROMPT = """You are reconciling several machine OCR transcriptions of ONE column \
from a 1910s Swedish printed tax directory (Antiqua type). Each entry is: \
Surname, initials., occupation, location, municipal-income–state-income. \
Below are {n} candidate transcriptions. Using the IMAGE as ground truth, output the single \
most accurate transcription of the column.

RULES: Preserve EVERY comma (the field delimiter) and EVERY en-dash between income figures. \
Keep one entry per line. Preserve Swedish characters å ä ö. Fix only characters that are \
clearly wrong versus the image. Do NOT paraphrase, reorder, translate, or add commentary. \
Output ONLY the reconciled transcription.

CANDIDATES:
{cands}
"""


def crop_for(pid: str, year: str) -> str:
    p = PROJ / f"Outputs/regions_{year}" / f"{pid}.png"
    return str(p)


def read_engine(eng: str, pid: str) -> str:
    f = VLM_RAW / f"{pid}__{eng}.txt"
    if f.exists():
        return bakeoff.clean(f.read_text(encoding="utf-8", errors="replace"))
    f2 = READS_EVAL / eng / f"{pid}.txt"
    if f2.exists():
        return bakeoff.clean(f2.read_text(encoding="utf-8", errors="replace"))
    return ""


def mode_vlm(mediator: str, voters: list[str]):
    from http_client import http_client
    ids = bakeoff.eval_ids()
    n_ok = 0
    for year, pid in ids:
        outp = VLM_RAW / f"{pid}__mediate-{mediator}.txt"
        if outp.exists() and outp.stat().st_size > 0:
            n_ok += 1; continue
        cands = []
        for i, v in enumerate(voters, 1):
            t = read_engine(v, pid)
            if t.strip():
                cands.append(f"[Candidate {i}]\n{t.strip()}")
        if len(cands) < 2:
            continue
        prompt = MEDIATE_PROMPT.format(n=len(cands), cands="\n\n".join(cands))
        try:
            text, _ = http_client(ENDPOINT, crop_for(pid, year), prompt)
            outp.write_text(text or "", encoding="utf-8"); n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {pid}: {type(e).__name__}: {str(e)[:80]}")
    print(f"[mediate-vlm {mediator}] wrote {n_ok} -> {VLM_RAW}")


def mode_ce(voters: list[str]):
    import consensus_entropy as ce
    ids = bakeoff.eval_ids()
    n = 0
    for year, pid in ids:
        cols = [read_engine(v, pid) for v in voters]
        cols = [c for c in cols if c.strip()]
        if len(cols) < 2:
            continue
        try:
            consensus, _prof, _res = ce.ce_consensus_cluster(cols)
        except Exception:
            consensus = max(cols, key=len)
        (VLM_RAW / f"{pid}__mediate-ce.txt").write_text(consensus or "", encoding="utf-8")
        n += 1
    print(f"[mediate-ce] wrote {n} -> {VLM_RAW}")


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vlm", "ce"], required=True)
    ap.add_argument("--mediator", default="Qwen3-VL-8B")
    ap.add_argument("--voters", default="dots.ocr,PaddleOCR-VL-1.6,tesseract")
    args = ap.parse_args(argv)
    voters = [v.strip() for v in args.voters.split(",") if v.strip()]
    if args.mode == "vlm":
        mode_vlm(args.mediator, voters)
    else:
        mode_ce(voters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
