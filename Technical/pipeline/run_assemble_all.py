#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_assemble_all.py -- Stage 3+6 capstone over the full corpus (CPU-only).

For every page, gather each engine's read of each region (header/colL/colR) from
Outputs/reads/<engine>/<page>_<region>.txt, fuse the engines GOLD-FREE via
s3_fuse.fuse_region (per-line Consensus-Entropy agreement = confidence), then
assemble per-page JSON + the master readable markdown + the separate page index,
plus a flagged-lines CSV (low cross-engine agreement = the gold-free review queue).

No model, no GPU, no network. Idempotent (recomputes from reads each run).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
import csv, glob, json, sys, unicodedata
from difflib import SequenceMatcher
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import s6_assemble  # noqa: E402

PROJ = _HERE.parent.parent
READS = PROJ / "Outputs" / "reads"
OUTD = PROJ / "Outputs" / "Data"
OUTD.mkdir(parents=True, exist_ok=True)
PAGEJSON = OUTD / "pages"
PAGEJSON.mkdir(exist_ok=True)

# ANCHOR = the canonical transcription (the VLM champion, full-corpus, cleanest).
# The other engines are CORROBORATORS: they supply a gold-free per-line confidence
# (does an independent engine read the same thing?) but never contribute lines --
# this avoids degrading the anchor with weaker engines' noise (No-Free-Lunch).
ANCHOR = "dots.ocr"
CORROBORATORS = ["olmOCR", "tesseract", "paddle", "kraken"]
REGIONS = ["header", "colL", "colR"]
LOW_CONF = 0.60   # mean corroborator similarity below this -> flag for review
CORROB_OK = 0.80  # a corroborator "confirms" a line at/above this similarity
WINDOW = 10       # search ±WINDOW lines around the proportional position


def read_engine(engine: str, page: str, region: str) -> str:
    p = READS / engine / f"{page}_{region}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    return ""


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", " ".join((s or "").split())).casefold()


def _lines(t: str) -> list:
    return [ln for ln in (t or "").splitlines() if ln.strip()]


def anchored_fuse(anchor_text: str, others: dict) -> dict:
    """Anchor on anchor_text; per anchor line, agreement = mean over corroborator
    engines of the best line-similarity (searched in a local window). Returns the
    anchor text verbatim as fused_text plus per-line confidence + corroboration."""
    a_lines = _lines(anchor_text)
    o_norm = {e: [_norm(x) for x in _lines(t)] for e, t in others.items() if t and t.strip()}
    out_lines = []
    agrs = []
    n = len(a_lines)
    for i, ln in enumerate(a_lines):
        an = _norm(ln)
        sims = {}
        corro = 0
        for e, ol in o_norm.items():
            if not ol:
                continue
            lo = max(0, int(i / max(1, n) * len(ol)) - WINDOW)
            hi = min(len(ol), lo + 2 * WINDOW + 1)
            best = 0.0
            for cand in ol[lo:hi]:
                r = SequenceMatcher(None, an, cand).ratio()
                if r > best:
                    best = r
            sims[e] = round(best, 3)
            if best >= CORROB_OK:
                corro += 1
        agr = round(sum(sims.values()) / len(sims), 4) if sims else None
        if agr is not None:
            agrs.append(agr)
        out_lines.append({"text": ln, "agreement": agr,
                          "n_corrob": corro, "sims": sims})
    mean_agr = round(sum(agrs) / len(agrs), 4) if agrs else None
    return {"fused_text": "\n".join(a_lines),
            "lines": out_lines, "mean_agreement": mean_agr,
            "n_corroborators": len(o_norm)}


def all_pages() -> list:
    ids = set()
    d = READS / ANCHOR
    if d.is_dir():
        for f in d.glob("SSA_*_col*.txt"):
            ids.add(f.stem.rsplit("_", 1)[0])  # SSA_0002
    return sorted(ids)


def main() -> int:
    pages = all_pages()
    print(f"[assemble] {len(pages)} pages with an anchor ({ANCHOR}) column read")
    records = []
    flagged = []  # (page, region, line_idx, agreement, n_corrob, text)
    cov = {ANCHOR: 0}
    cov.update({e: 0 for e in CORROBORATORS})
    multi = 0
    for page in pages:
        page_read = {"ssa_id": page}
        for region in REGIONS:
            anchor_t = read_engine(ANCHOR, page, region)
            if anchor_t and anchor_t.strip():
                cov[ANCHOR] += 1
            others = {}
            for e in CORROBORATORS:
                t = read_engine(e, page, region)
                if t and t.strip():
                    others[e] = t
                    cov[e] += 1
            if not (anchor_t and anchor_t.strip()):
                # no anchor: fall back to the best available corroborator verbatim
                fb = next(iter(others.values()), "")
                page_read[region] = {"text": fb, "confidence": None, "n_corrob": 0}
                continue
            fr = anchored_fuse(anchor_t, others)
            if fr["n_corroborators"] >= 1 and region in ("colL", "colR"):
                multi += 1
            page_read[region] = {"text": fr["fused_text"],
                                 "confidence": fr["mean_agreement"],
                                 "n_corrob": fr["n_corroborators"]}
            for i, ln in enumerate(fr["lines"]):
                if ln["agreement"] is not None and ln["agreement"] < LOW_CONF:
                    flagged.append((page, region, i, ln["agreement"],
                                    ln["n_corrob"], ln["text"][:200]))
        rec = s6_assemble.assemble_page(page_read)
        rec["_corrob"] = {r: page_read.get(r, {}).get("n_corrob", 0)
                          for r in ("colL", "colR")}
        records.append(rec)
        s6_assemble.write_page_json(rec, str(PAGEJSON))

    # master readable + page index
    master = OUTD / "kalendern_1936_readable.md"
    s6_assemble.write_master(records, str(master))
    idx = OUTD / "page_index.csv"
    s6_assemble.write_page_index(records, str(idx))
    # flagged review queue
    fq = OUTD / "flagged_lines.csv"
    with fq.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "region", "line_idx", "agreement", "n_votes", "text"])
        w.writerows(flagged)

    print(f"[assemble] engine coverage (regions): " +
          " ".join(f"{e}={cov[e]}" for e in [ANCHOR] + CORROBORATORS))
    print(f"[assemble] corroborated column regions: {multi}")
    print(f"[assemble] flagged low-agreement lines: {len(flagged)}")
    print(f"[assemble] wrote:\n  {master}\n  {idx}\n  {fq}\n  {PAGEJSON}/ ({len(records)} json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
