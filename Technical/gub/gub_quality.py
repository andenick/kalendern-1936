#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gub_quality.py -- Phase 1 quality + structure research over a GUB volume's text layer.

Part A (this script, CPU, all pages): page classification, gutter-detection success,
ad-contamination, income-dash cleanliness, and alphabetical reading-order monotonicity,
from the coordinate column split (gub_columns.split_page). Writes a per-volume section
into Outputs/Reports/KALENDERN_GUB_QUALITY_REPORT.md + a per-page CSV.

Part B (dots.ocr re-OCR agreement on a sample) is gub_quality_sample.py (GPU).

  python gub_quality.py --volume taxkal_1912
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, csv, glob, json, re, statistics as st, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import gub_columns as gc  # noqa: E402

PROJ = _HERE.parent.parent
DASHES = "–—―‒-"
_CLEAN_INCOME = re.compile(r"\d[\d ]*[–—]\s*\d")          # em/en dash between digits
_DIRTY_INCOME = re.compile(r"\d{3,}[ \-]+\d{3,}")          # space/hyphen-separated pair
_SURNAME = re.compile(r"^\s*([A-ZÅÄÖ][A-Za-zÅÄÖåäö:\-]+)\s*,")


def page_stats(rec: dict) -> dict:
    res = gc.split_page(rec)
    n_entry = sum(res["columns"][c]["n_entry"] for c in ("colL", "colR"))
    n_lines = sum(res["columns"][c]["n_lines"] for c in ("colL", "colR"))
    n_ad = sum(res["columns"][c]["n_ad"] for c in ("colL", "colR"))
    clean = dirty = 0
    surns = []
    for c in ("colL", "colR"):
        for ln, tg in zip(res["columns"][c]["lines"], res["columns"][c]["tags"]):
            if tg == "entry":
                if _CLEAN_INCOME.search(ln):
                    clean += 1
                elif _DIRTY_INCOME.search(ln):
                    dirty += 1
                m = _SURNAME.match(ln)
                if m:
                    surns.append(m.group(1))
    entry_ratio = n_entry / n_lines if n_lines else 0.0
    is_body = n_entry >= 20 and entry_ratio >= 0.45
    return {"page": rec["page"], "n_words": len(rec["words"]),
            "gutter_conf": res["gutter_conf"], "n_lines": n_lines,
            "n_entry": n_entry, "entry_ratio": round(entry_ratio, 3),
            "n_ad": n_ad, "income_clean": clean, "income_dirty": dirty,
            "is_body": int(is_body), "surnames": surns}


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    args = ap.parse_args(argv)
    reg = json.loads((PROJ / "Technical/volumes.json").read_text(encoding="utf-8"))
    vol = next(v for v in reg["volumes"] if v["volume_id"] == args.volume)
    year = vol["year"]
    wdir = PROJ / f"Technical/gub/words_{year}"
    files = sorted(glob.glob(str(wdir / "p*.json")))

    rows = []
    for f in files:
        rows.append(page_stats(json.loads(Path(f).read_text(encoding="utf-8"))))

    body = [r for r in rows if r["is_body"]]
    # gutter success on body pages
    gconf = [r["gutter_conf"] for r in body]
    gutter_ok = sum(c >= 0.5 for c in gconf)
    # ad contamination on body pages
    tot_lines = sum(r["n_lines"] for r in body) or 1
    tot_ad = sum(r["n_ad"] for r in body)
    # income dash cleanliness
    cl = sum(r["income_clean"] for r in body); dt = sum(r["income_dirty"] for r in body)
    dash_clean = cl / (cl + dt) if (cl + dt) else 0.0
    # alphabetical monotonicity across body pages (in page order)
    surns = [s for r in body for s in r["surnames"]]
    asc = sum(1 for a, b in zip(surns, surns[1:]) if a.casefold() <= b.casefold())
    mono = asc / (len(surns) - 1) if len(surns) > 1 else 0.0
    tot_entry = sum(r["n_entry"] for r in body)

    summary = {
        "volume": args.volume, "year": year, "total_pages": len(rows),
        "body_pages": len(body), "nonbody_pages": len(rows) - len(body),
        "mean_gutter_conf": round(st.mean(gconf), 3) if gconf else 0,
        "gutter_ok_pct": round(100 * gutter_ok / len(body), 1) if body else 0,
        "entries_total": tot_entry,
        "mean_entries_per_body_page": round(tot_entry / len(body), 1) if body else 0,
        "ad_line_pct_body": round(100 * tot_ad / tot_lines, 2),
        "income_dash_clean_pct": round(100 * dash_clean, 1),
        "surname_alpha_monotonicity": round(mono, 3),
    }

    # write per-page CSV
    outd = PROJ / "Outputs" / "Reports"
    outd.mkdir(parents=True, exist_ok=True)
    with (outd / f"gub_quality_{year}_pages.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["page", "n_words", "gutter_conf", "n_lines", "n_entry",
                    "entry_ratio", "n_ad", "income_clean", "income_dirty", "is_body"])
        for r in rows:
            w.writerow([r[k] for k in ("page", "n_words", "gutter_conf", "n_lines",
                        "n_entry", "entry_ratio", "n_ad", "income_clean",
                        "income_dirty", "is_body")])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (outd / f"gub_quality_{year}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
