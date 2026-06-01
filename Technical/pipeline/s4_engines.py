#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalendern reading pipeline -- Stage 4 (engine voters): Tesseract-swe + Kraken.

CPU-ONLY. Produces additional OCR "votes" for the Stage-3 fusion by running two
independent recognizers over each column crop:

  - **Tesseract** (system install, ``lang=swe``) via pytesseract.
  - **Kraken** (CPU) with a **CATMuS-Print** recognition model (historical print),
    run as a subprocess with ``PYTHONUTF8=1``.

For a page it uses the colL/colR crops in ``Outputs\\regions\\`` if present; if a
page hasn't been cropped yet, it crops it on the fly via ``s2_column_split``.

Outputs (one .txt per region per engine):
  Outputs\\reads\\tesseract\\<page>_<region>.txt
  Outputs\\reads\\kraken\\<page>_<region>.txt

Usage:
  python s4_engines.py --pages 30 --engines tesseract kraken
  python s4_engines.py --page-ids SSA_0001 SSA_0002 --engines tesseract
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import s2_column_split as s2  # noqa: E402

# --- project paths --------------------------------------------------------- #
import os
PROJ = Path(__file__).resolve().parents[2]  # repo root (.../Kalendern)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs"))
IMAGES = DATA_ROOT / "Inputs" / "Images"
REGIONS = OUTPUT_ROOT / "regions"
READS = OUTPUT_ROOT / "reads"

# --- system tool locations ------------------------------------------------- #
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
KRAKEN_EXE = str(
    PROJ / "Technical" / "experiments" / "kraken_venv" / "Scripts" / "kraken.exe"
)
KRAKEN_MODELS = PROJ / "Technical" / "experiments" / "kraken_models"


# --------------------------------------------------------------------------- #
# crop access
# --------------------------------------------------------------------------- #
def ensure_crops(page_id: str) -> dict:
    """Return {colL, colR} crop paths for a page, cropping via s2 if missing."""
    colL = REGIONS / f"{page_id}_colL.png"
    colR = REGIONS / f"{page_id}_colR.png"
    if colL.exists() and colR.exists():
        return {"colL": str(colL), "colR": str(colR)}
    img = IMAGES / f"{page_id}.jpg"
    if not img.exists():
        raise FileNotFoundError(f"no input image for {page_id}: {img}")
    s2.process_image(str(img), str(REGIONS), bin_method="sauvola")
    return {"colL": str(colL), "colR": str(colR)}


# --------------------------------------------------------------------------- #
# Tesseract-swe
# --------------------------------------------------------------------------- #
def run_tesseract(crop_path: str, lang: str = "swe", psm: int = 4) -> str:
    """OCR one crop with system Tesseract (Swedish). psm 4 = single column of
    text of variable sizes -- right for a directory column."""
    try:
        import pytesseract
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"pytesseract/PIL not available: {e}")
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    config = f"--oem 1 --psm {psm}"
    img = Image.open(crop_path)
    return pytesseract.image_to_string(img, lang=lang, config=config)


# --------------------------------------------------------------------------- #
# Kraken + CATMuS
# --------------------------------------------------------------------------- #
def _find_kraken_model() -> Optional[str]:
    """Locate a downloaded CATMuS-Print (or any *.mlmodel) recognition model."""
    if not KRAKEN_MODELS.exists():
        return None
    # prefer a CATMuS print model by name; else first .mlmodel
    cands = list(KRAKEN_MODELS.glob("*.mlmodel"))
    if not cands:
        return None
    for c in cands:
        if "catmus" in c.name.lower() or "print" in c.name.lower():
            return str(c)
    return str(cands[0])


def run_kraken(crop_path: str, model_path: Optional[str] = None,
               timeout: int = 600) -> str:
    """OCR one crop with Kraken (binarize -> segment -> recognize) on CPU.

    Runs the kraken CLI as a subprocess with PYTHONUTF8=1 so Windows console
    encoding never corrupts å/ä/ö. Uses the bundled blla segmenter + the CATMuS
    recognition model."""
    model = model_path or _find_kraken_model()
    if model is None:
        raise RuntimeError(
            f"no kraken recognition model under {KRAKEN_MODELS} "
            f"(run: kraken get 10.5281/zenodo.10592716)"
        )
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
    out_txt = crop_path + ".kraken.txt"
    # kraken -i <in> <out> binarize segment ocr -m <model>
    cmd = [
        KRAKEN_EXE, "-i", crop_path, out_txt,
        "binarize", "segment", "ocr", "-m", model,
    ]
    try:
        subprocess.run(cmd, env=env, check=True, timeout=timeout,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"kraken failed: {e.stderr[-500:] if e.stderr else e}")
    p = Path(out_txt)
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass
    return text


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def page_ids_first(n: int) -> List[str]:
    imgs = sorted(IMAGES.glob("SSA_*.jpg"))
    return [p.stem for p in imgs[:n]]


def run_page(page_id: str, engines: List[str],
             kraken_model: Optional[str] = None) -> dict:
    crops = ensure_crops(page_id)
    results = {}
    for region in ("colL", "colR"):
        crop = crops[region]
        for eng in engines:
            out_dir = READS / eng
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{page_id}_{region}.txt"
            try:
                if eng == "tesseract":
                    text = run_tesseract(crop)
                elif eng == "kraken":
                    text = run_kraken(crop, kraken_model)
                else:
                    raise ValueError(f"unknown engine {eng}")
                out_path.write_text(text, encoding="utf-8")
                results[f"{region}/{eng}"] = ("ok", len(text))
            except Exception as e:  # noqa: BLE001
                results[f"{region}/{eng}"] = ("ERROR", str(e)[:160])
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Kalendern Stage-4 engine voters")
    ap.add_argument("--pages", type=int, default=30,
                    help="run the first N pages")
    ap.add_argument("--page-ids", nargs="*", default=None,
                    help="explicit page ids (overrides --pages)")
    ap.add_argument("--engines", nargs="+", default=["tesseract", "kraken"],
                    choices=["tesseract", "kraken"])
    ap.add_argument("--kraken-model", default=None)
    args = ap.parse_args()

    ids = args.page_ids or page_ids_first(args.pages)
    print(f"[S4] {len(ids)} pages, engines={args.engines}")
    for pid in ids:
        try:
            res = run_page(pid, args.engines, args.kraken_model)
            summary = "  ".join(f"{k}={v[0]}({v[1]})" for k, v in res.items())
            print(f"[S4] {pid}: {summary}")
        except Exception as e:  # noqa: BLE001
            print(f"[S4] {pid}: ERROR {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
