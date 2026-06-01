#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_classical.py -- run classical OCR engines over the bake-off eval column crops
(CPU), writing to Outputs/reads_eval/<engine>/<page_id>.txt for the leaderboard.

Engines: tesseract-swe (pytesseract, psm 4 = single column) now; Kraken/PaddleOCR-sv
pluggable (each lives in its own venv). Reads the frozen eval manifest.

  python run_classical.py --volume taxkal_1912 --engines tesseract
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, csv, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "pipeline"))
PROJ = _HERE.parent.parent
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def run_tesseract(crop_path: str, lang="swe", psm=4) -> str:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = TESS
    return pytesseract.image_to_string(Image.open(crop_path), lang=lang,
                                       config=f"--oem 1 --psm {psm}")


ENGINES = {"tesseract": run_tesseract}


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--engines", nargs="+", default=["tesseract"])
    args = ap.parse_args(argv)
    year = args.volume.split("_")[-1]
    man = _HERE / f"sample_{year}_manifest.csv"
    rows = list(csv.DictReader(man.open(encoding="utf-8")))
    for eng in args.engines:
        fn = ENGINES.get(eng)
        if not fn:
            print(f"  skip unknown engine {eng}"); continue
        outd = PROJ / "Outputs" / "reads_eval" / eng
        outd.mkdir(parents=True, exist_ok=True)
        n = 0; t0 = time.time()
        for r in rows:
            outp = outd / f"{r['page_id']}.txt"
            if outp.exists() and outp.stat().st_size > 0:
                continue
            try:
                outp.write_text(fn(r["image_path"]) or "", encoding="utf-8"); n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {r['page_id']}: {type(e).__name__}: {str(e)[:80]}")
        print(f"[classical {eng}] {args.volume}: wrote {n} ({time.time()-t0:.0f}s) -> {outd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
