#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_gub_dataset.py -- Phase 3: build (column-image -> silver-text) training pairs
from high-confidence GUB body pages, and a gold-registry manifest.

Selection: body pages with high gutter confidence, high entry ratio, low ad rate (from
gub_quality_<year>_pages.csv). For each, render the page image, coordinate-split, crop the
two columns, and write the GUB column body (ad-filtered, reading-order-rebuilt) as the label.

Outputs:
  Technical/gub/dataset_<year>/<page_id>.png            -- column crop image
  Technical/gub/dataset_<year>/<page_id>.txt            -- silver reference text
  Grace .../gold/sv_taxkal_<year>_manifest.csv          -- gold-registry manifest

  python build_gub_dataset.py --volume taxkal_1912 --n 100
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, csv, json, subprocess, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import gub_columns as gc  # noqa: E402
from PIL import Image     # noqa: E402

PROJ = _HERE.parent.parent
GRACE_GOLD = Path(os.environ.get("GOLD_DIR", "data/gold"))
NATIVE_PY = os.environ.get("NATIVE_PY", sys.executable)


def select_pages(year: int, n: int, min_conf: float, min_entry: int, max_ad: int) -> list[int]:
    csvp = PROJ / "Outputs" / "Reports" / f"gub_quality_{year}_pages.csv"
    good = []
    for r in csv.DictReader(csvp.open(encoding="utf-8")):
        if (r["is_body"] == "1" and float(r["gutter_conf"]) >= min_conf
                and int(r["n_entry"]) >= min_entry and int(r["n_ad"]) <= max_ad
                and float(r["entry_ratio"]) >= 0.6):
            good.append(int(r["page"]))
    if len(good) <= n:
        return good
    step = len(good) / n
    return [good[int(i * step)] for i in range(n)]


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--min-entry", type=int, default=45)
    ap.add_argument("--max-ad", type=int, default=4)
    args = ap.parse_args(argv)
    reg = json.loads((PROJ / "Technical/volumes.json").read_text(encoding="utf-8"))
    vol = next(v for v in reg["volumes"] if v["volume_id"] == args.volume)
    year = vol["year"]
    pages = select_pages(year, args.n, args.min_conf, args.min_entry, args.max_ad)
    print(f"[dataset] {args.volume}: {len(pages)} pages selected")

    # render the page images we need
    subprocess.run([NATIVE_PY, str(_HERE / "gub_extract.py"), "--volume", args.volume,
                    "--pages", ",".join(str(p) for p in pages)], check=True)

    dsdir = _HERE / f"dataset_{year}"; dsdir.mkdir(parents=True, exist_ok=True)
    imgdir = PROJ / f"Inputs/Images_{year}"
    wdir = PROJ / f"Technical/gub/words_{year}"
    rows = []
    n_pairs = 0
    for p in pages:
        rec = json.loads((wdir / f"p{p:04d}.json").read_text(encoding="utf-8"))
        res = gc.split_page(rec)
        img = Image.open(imgdir / f"p{p:04d}.png")
        W, H = img.size
        sx = int(res["split_x"] * W / rec["rect"][0])
        for col, box in (("colL", (0, 0, sx, H)), ("colR", (sx, 0, W, H))):
            ref = res["columns"][col]["body"].strip()
            if len(ref) < 80:            # skip near-empty columns
                continue
            pid = f"{year}_p{p:04d}_{col}"
            cpath = dsdir / f"{pid}.png"
            img.crop(box).save(cpath)
            (dsdir / f"{pid}.txt").write_text(ref, encoding="utf-8")
            rows.append({
                "page_id": pid, "image_path": str(cpath), "reference": ref,
                "stratum": f"sv_taxkal_{year}", "metric": "cer",
                "gold_path": str(dsdir / f"{pid}.txt"), "status": "silver_ocr",
                "source": "GUB digitisation OCR (coordinate-split, ad-filtered)",
                "license": "GUB digitisation", "verified": "false", "language": "sv",
                "region": "print_column", "notes": f"Taxeringskalender {year}",
                "content_bearing": "true"})
            n_pairs += 1

    GRACE_GOLD.mkdir(parents=True, exist_ok=True)
    man = GRACE_GOLD / f"sv_taxkal_{year}_manifest.csv"
    cols = ["page_id", "image_path", "reference", "stratum", "metric", "gold_path",
            "status", "source", "license", "verified", "language", "region", "notes",
            "content_bearing"]
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[dataset] {n_pairs} (image,text) pairs -> {dsdir}")
    print(f"[dataset] manifest -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
