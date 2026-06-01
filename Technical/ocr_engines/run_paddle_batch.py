#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overnight batch PaddleOCR-sv runner (CPU-only).

Loads the PaddleOCR Swedish/Latin model ONCE and OCRs every region image
(_colL.png, _colR.png, _header.png) in the regions dir, writing per-region
text files to <reads>/paddle/<page>_<region>.txt.

CRITICAL: enable_mkldnn=False (avoids a Windows crash). CPU device.

Idempotent: skips a region whose output .txt already exists and is non-empty.

At the end (or each --flush-every N), rewrites paddle_manifest.csv with
(page, region, text_path, n_chars, n_lines) for every existing output.

Usage:
  python run_paddle_batch.py --regions <regions> --reads <reads>
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["FLAGS_use_cuda"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys
import csv
import glob
import argparse
import traceback


REGIONS = ["colL", "colR", "header"]


def region_inputs(regions_dir):
    """Yield (page_id, region, img_path) for all region images present."""
    items = []
    for reg in REGIONS:
        for p in sorted(glob.glob(os.path.join(regions_dir, f"*_{reg}.png"))):
            base = os.path.basename(p)
            # page id = filename minus "_<region>.png"
            page_id = base[: -(len(reg) + 5)]  # strip "_<reg>.png"
            items.append((page_id, reg, p))
    return items


def out_path(reads_dir, page_id, region):
    return os.path.join(reads_dir, "paddle", f"{page_id}_{region}.txt")


def is_done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def build_manifest(reads_dir):
    paddle_dir = os.path.join(reads_dir, "paddle")
    rows = []
    for p in sorted(glob.glob(os.path.join(paddle_dir, "*.txt"))):
        base = os.path.basename(p)[:-4]  # strip .txt
        # region is the last underscore-token
        region = base.rsplit("_", 1)[-1]
        page = base[: -(len(region) + 1)]
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            txt = ""
        n_chars = len(txt)
        n_lines = txt.count("\n") + (1 if txt and not txt.endswith("\n") else 0)
        if txt == "":
            n_lines = 0
        rows.append((page, region, p, n_chars, n_lines))
    man = os.path.join(reads_dir, "paddle_manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "region", "text_path", "n_chars", "n_lines"])
        for r in rows:
            w.writerow(r)
    return man, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.reads, "paddle"), exist_ok=True)

    items = region_inputs(args.regions)
    todo = [it for it in items
            if not is_done(out_path(args.reads, it[0], it[1]))]
    print(f"[PADDLE] regions found={len(items)} todo={len(todo)} "
          f"(skipping {len(items) - len(todo)} existing)", flush=True)

    if todo:
        from paddleocr import PaddleOCR
        # Detection: PP-OCRv5 *mobile* detector at native resolution — ~2x
        # faster than server_det on CPU for these tall dense columns with
        # identical line yield/accuracy on this corpus (benchmarked). A
        # det side-length limit was rejected: downscaling the small Antiqua
        # print fragments detection and injects noise.
        # Recognition: latin_PP-OCRv5_mobile_rec — the diacritic-accurate
        # Swedish/Latin recogniser (same family selected as best for å/ä/ö).
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu", enable_mkldnn=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        )
        print("[PADDLE] model loaded (mobile_det + latin_mobile_rec)",
              flush=True)

        done = 0
        errors = 0
        for page_id, region, img in todo:
            outp = out_path(args.reads, page_id, region)
            try:
                result = ocr.predict(img)
                lines = []
                for res in result:
                    for t in res.get("rec_texts", []):
                        lines.append(t)
                text = "\n".join(lines)
                # write even if empty (marks attempted); but empty stays
                # eligible for re-run since is_done checks size>0
                with open(outp, "w", encoding="utf-8") as f:
                    f.write(text)
                done += 1
                if done % 25 == 0:
                    print(f"[PADDLE] {done}/{len(todo)} last={page_id}_{region} "
                          f"lines={len(lines)} chars={len(text)}", flush=True)
                if args.flush_every and done % args.flush_every == 0:
                    build_manifest(args.reads)
            except Exception as e:
                errors += 1
                print(f"[PADDLE] {page_id}_{region}: ERROR {e}",
                      file=sys.stderr, flush=True)
                traceback.print_exc()
        print(f"[PADDLE] OCR loop done: ok={done} errors={errors}", flush=True)

    man, n = build_manifest(args.reads)
    print(f"[PADDLE] manifest -> {man} ({n} rows)", flush=True)


if __name__ == "__main__":
    main()
