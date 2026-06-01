#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bakeoff.py -- score every OCR engine's reads of the frozen eval set against the
GUB reference and emit a leaderboard (CER + comma/dash/field-integrity metrics).

Inputs (per volume year):
  refs:      Outputs/reads_gub/<year>/<page_id>.txt        (GUB coordinate-split body)
  VLM reads: Outputs/reads_eval/_vlm/<page_id>__<model>.txt (from race_e2e --outdir)
  classical: Outputs/reads_eval/<engine>/<page_id>.txt      (from run_classical.py)

Output: Outputs/Reports/BAKEOFF_LEADERBOARD.md + bakeoff_scores.csv
Idempotent / re-runnable as more engine reads land. CPU only.
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTHONUTF8", "1")
import csv, glob, re, statistics as st, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import punct_scorer as ps  # noqa: E402

PROJ = _HERE.parent.parent
REPORTS = PROJ / "Outputs" / "Reports"
VLM_RAW = PROJ / "Outputs" / "reads_eval" / "_vlm"
READS_EVAL = PROJ / "Outputs" / "reads_eval"
YEARS = ["1912", "1914"]

_YAML = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.S)
_TAG = re.compile(r"<[^>]+>")
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```\s*$", re.I)


def clean(t: str) -> str:
    t = _YAML.sub("", t or "")                  # olmOCR YAML front-matter
    t = _FENCE.sub("", t)
    t = _TAG.sub(" ", t)                         # dots.ocr HTML tables, etc.
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return t


def ref_for(year: str, pid: str) -> str:
    p = PROJ / "Outputs" / "reads_gub" / year / f"{pid}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def discover():
    """engine -> {page_id: read_path}."""
    eng = {}
    for f in glob.glob(str(VLM_RAW / "*__*.txt")):
        b = Path(f).name
        pid, model = b[:-4].split("__", 1)
        eng.setdefault(model, {})[pid] = f
    for d in READS_EVAL.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            for f in d.glob("*.txt"):
                eng.setdefault(d.name, {})[f.stem] = str(f)
    return eng


def eval_ids() -> list[tuple[str, str]]:
    ids = []
    for y in YEARS:
        man = _HERE / f"sample_{y}_manifest.csv"
        if man.exists():
            for r in csv.DictReader(man.open(encoding="utf-8")):
                ids.append((y, r["page_id"]))
    return ids


def main() -> int:
    ids = eval_ids()
    engines = discover()
    rows = []
    for model, pages in engines.items():
        ms = {k: [] for k in ("cer", "comma_f1", "comma_recall", "dash_recall",
                              "digit_recall", "field_count_ratio")}
        n = 0
        for year, pid in ids:
            if pid not in pages:
                continue
            ref = ref_for(year, pid)
            if len(ref.strip()) < 40:
                continue
            hyp = clean(Path(pages[pid]).read_text(encoding="utf-8", errors="replace"))
            sc = ps.score(ref, hyp)
            for k in ms:
                ms[k].append(sc[k])
            n += 1
        if n:
            rows.append({"engine": model, "n": n,
                         **{k: round(st.mean(v), 4) for k, v in ms.items()}})
    # composite: low CER + high comma_f1 + high field integrity
    for r in rows:
        r["score"] = round((1 - r["cer"]) * 0.4 + r["comma_f1"] * 0.35
                           + r["field_count_ratio"] * 0.25, 4)
    rows.sort(key=lambda r: -r["score"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    cols = ["engine", "n", "score", "cer", "comma_f1", "comma_recall",
            "dash_recall", "digit_recall", "field_count_ratio"]
    with (REPORTS / "bakeoff_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    lines = ["# Kalendern OCR bake-off leaderboard (GUB-as-reference)", "",
             f"_Eval: {len(ids)} columns across {', '.join(YEARS)}. "
             "score = 0.40·(1−CER) + 0.35·comma_f1 + 0.25·field_integrity. "
             "Higher is better; comma_f1/field_count guard against dropped delimiters._", "",
             "| rank | engine | n | score | CER | comma_f1 | comma_rec | dash_rec | digit_rec | field_int |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | **{r['engine']}** | {r['n']} | {r['score']} | {r['cer']} | "
                     f"{r['comma_f1']} | {r['comma_recall']} | {r['dash_recall']} | "
                     f"{r['digit_recall']} | {r['field_count_ratio']} |")
    (REPORTS / "BAKEOFF_LEADERBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {REPORTS/'BAKEOFF_LEADERBOARD.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
