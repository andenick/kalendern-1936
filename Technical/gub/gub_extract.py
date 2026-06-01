#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gub_extract.py -- Phase 0 extraction primitive for GUB already-OCRed Kalendern PDFs.

For an `ocr_pdf` volume in volumes.json, extracts per page:
  - the embedded OCR text layer as WORDS with coordinates (x0,y0,x1,y1,text, block,line,wno)
  - span geometry (font size proxy via height) for ad/banner detection downstream
  - a rendered page image (the scanned page) at a target DPI for re-OCR / training

Outputs (idempotent):
  Inputs/Images_<year>/p<NNNN>.png                 -- rendered page image
  Technical/gub/words_<year>/p<NNNN>.json          -- {page, rect, words:[...], spans:[...]}

CPU-only, read-only on the PDF. Uses PyMuPDF (installed in .venv-native).

  python gub_extract.py --volume taxkal_1912 [--dpi 300] [--limit N] [--pages a,b,c]
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, json, sys
from pathlib import Path

import fitz  # PyMuPDF

PROJ = Path(__file__).resolve().parent.parent.parent
REG = PROJ / "Technical" / "volumes.json"


def load_volume(volume_id: str) -> dict:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    for v in reg["volumes"]:
        if v["volume_id"] == volume_id:
            return v
    raise SystemExit(f"unknown volume {volume_id}; known: "
                     + ", ".join(v['volume_id'] for v in reg['volumes']))


def page_words(page) -> list[dict]:
    """Word tuples (x0,y0,x1,y1,word,block,line,wno) -> list of dicts."""
    out = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, txt, b, l, wn = w
        if txt.strip():
            out.append({"x0": round(x0, 1), "y0": round(y0, 1),
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "t": txt, "b": b, "l": l, "wn": wn})
    return out


def page_spans(page) -> list[dict]:
    """Per-span geometry (size = font-size proxy) for banner/ad detection.

    Uses dict mode (not rawdict): dict spans carry 'text'/'size'/'bbox';
    rawdict spans carry per-char dicts under 'chars' with no 'text' key.
    """
    out = []
    raw = page.get_text("dict")
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                bb = sp.get("bbox")
                txt = sp.get("text", "")
                if bb and txt.strip():
                    out.append({"x0": round(bb[0], 1), "y0": round(bb[1], 1),
                                "x1": round(bb[2], 1), "y1": round(bb[3], 1),
                                "size": round(sp.get("size", 0), 1),
                                "t": txt})
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only (0=all)")
    ap.add_argument("--pages", default="", help="comma list of 0-based page indices")
    ap.add_argument("--no-images", action="store_true", help="skip image render (words only)")
    args = ap.parse_args(argv)

    vol = load_volume(args.volume)
    if vol["kind"] != "ocr_pdf":
        raise SystemExit(f"{args.volume} is kind={vol['kind']}, not ocr_pdf")
    year = vol["year"]
    pdf = PROJ / vol["source_pdf"]
    imgdir = PROJ / f"Inputs/Images_{year}"
    worddir = PROJ / f"Technical/gub/words_{year}"
    imgdir.mkdir(parents=True, exist_ok=True)
    worddir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf))
    if args.pages:
        idxs = [int(x) for x in args.pages.split(",") if x.strip()]
    else:
        idxs = list(range(doc.page_count if not args.limit else min(args.limit, doc.page_count)))

    zoom = args.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    n_w = n_i = n_skip = 0
    for i in idxs:
        page = doc.load_page(i)
        wp = worddir / f"p{i:04d}.json"
        ip = imgdir / f"p{i:04d}.png"
        if wp.exists() and (args.no_images or ip.exists()):
            n_skip += 1
            continue
        rect = page.rect
        rec = {"page": i, "year": year, "volume": args.volume,
               "rect": [round(rect.width, 1), round(rect.height, 1)],
               "words": page_words(page), "spans": page_spans(page)}
        wp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        n_w += 1
        if not args.no_images and not ip.exists():
            page.get_pixmap(matrix=mat).save(str(ip))
            n_i += 1
        if (n_w + n_skip) % 100 == 0:
            print(f"  ... {n_w} words-json, {n_i} images, {n_skip} skip", flush=True)
    doc.close()
    print(f"[gub_extract {args.volume}] pages={len(idxs)} words_json={n_w} images={n_i} "
          f"skip={n_skip} -> {worddir} , {imgdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
