#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gub_quality_sample.py -- Phase 1 Part B prep: crop column images + write GUB reference
text + a dots.ocr race manifest for a stratified body-page sample of a GUB volume.

After this, run dots.ocr over the manifest (serve_model.ps1 + race_e2e.py), then score
the dots.ocr reads vs the GUB references with score_gub_agreement.py.

  python gub_quality_sample.py --volume taxkal_1912 --n 10
Outputs:
  Outputs/regions_<year>/<year>_p<NNNN>_col[LR].png    -- column crops
  Outputs/reads_gub/<year>/<year>_p<NNNN>_col[LR].txt  -- GUB reference (column body)
  Technical/gub/sample_<year>_manifest.csv             -- race_e2e manifest
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
NATIVE_PY = os.environ.get("NATIVE_PY", sys.executable)


def stratified_body_pages(year: int, n: int) -> list[int]:
    csvp = PROJ / "Outputs" / "Reports" / f"gub_quality_{year}_pages.csv"
    body = [int(r["page"]) for r in csv.DictReader(csvp.open(encoding="utf-8"))
            if r["is_body"] == "1"]
    if not body:
        return []
    step = max(1, len(body) // n)
    return body[::step][:n]


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args(argv)
    reg = json.loads((PROJ / "Technical/volumes.json").read_text(encoding="utf-8"))
    vol = next(v for v in reg["volumes"] if v["volume_id"] == args.volume)
    year = vol["year"]
    pages = stratified_body_pages(year, args.n)
    print(f"[sample] {args.volume}: {len(pages)} pages -> {pages}")

    # ensure page images exist (render just these)
    subprocess.run([NATIVE_PY, str(_HERE / "gub_extract.py"), "--volume", args.volume,
                    "--pages", ",".join(str(p) for p in pages)], check=True)

    regdir = PROJ / f"Outputs/regions_{year}"; regdir.mkdir(parents=True, exist_ok=True)
    refdir = PROJ / f"Outputs/reads_gub/{year}"; refdir.mkdir(parents=True, exist_ok=True)
    imgdir = PROJ / f"Inputs/Images_{year}"
    wdir = PROJ / f"Technical/gub/words_{year}"

    rows = []
    for p in pages:
        rec = json.loads((wdir / f"p{p:04d}.json").read_text(encoding="utf-8"))
        res = gc.split_page(rec)
        img = Image.open(imgdir / f"p{p:04d}.png")
        W, H = img.size
        rect_w = rec["rect"][0]
        sx = int(res["split_x"] * W / rect_w)
        crops = {"colL": img.crop((0, 0, sx, H)), "colR": img.crop((sx, 0, W, H))}
        for col in ("colL", "colR"):
            pid = f"{year}_p{p:04d}_{col}"
            cpath = regdir / f"{pid}.png"
            crops[col].save(cpath)
            (refdir / f"{pid}.txt").write_text(res["columns"][col]["body"], encoding="utf-8")
            rows.append({"page_id": pid, "image_path": str(cpath),
                         "reference": "", "stratum": f"gub_{year}", "metric": "cer"})
    man = _HERE / f"sample_{year}_manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "image_path", "reference", "stratum", "metric"])
        w.writeheader(); w.writerows(rows)
    print(f"[sample] crops+refs for {len(rows)} columns; manifest -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
