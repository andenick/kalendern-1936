#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overnight batch driver: run S0 (preprocess) + S2 (column split) on ALL pages.

CPU-ONLY. Idempotent: a page is skipped if its core region outputs already
exist (_gray.png, _bin.png, _colL.png, _colR.png). Header is optional (some
pages legitimately have no header band).

Writes/updates a crop_failures.csv recording any page where:
  - the page errored entirely, OR
  - the gutter split was low-confidence ("poor"), OR
  - no header band was detected.

Usage:
  python run_crop_all.py --indir <imgdir> --outdir <regions>
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import sys
import csv
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_preprocess as s0
import s2_column_split as s2


CORE_SUFFIXES = ["_gray.png", "_bin.png", "_colL.png", "_colR.png"]


def page_done(outdir, page_id):
    return all(os.path.exists(os.path.join(outdir, page_id + suf))
               for suf in CORE_SUFFIXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bin", default="sauvola", choices=["sauvola", "otsu"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.indir, "*.jpg")))
    print(f"[CROP] {len(images)} input pages; outdir={args.outdir}", flush=True)

    fail_path = os.path.join(args.outdir, "crop_failures.csv")
    failures = []  # (page, stage, reason, detail)

    done = skipped = processed = errored = 0
    for img in images:
        page_id = os.path.splitext(os.path.basename(img))[0]
        if page_done(args.outdir, page_id):
            skipped += 1
            done += 1
            continue
        try:
            info, q = s2.process_image(img, args.outdir, bin_method=args.bin)
            processed += 1
            done += 1
            # flag low-confidence splits / missing headers
            if info["gutter_conf"] == "poor":
                failures.append((page_id, "gutter", "poor_confidence",
                                 f"ratio={info['gutter_diag'].get('gutter_ratio')} "
                                 f"split_frac={info['split_frac']}"))
            if "header" not in info["paths"]:
                failures.append((page_id, "header", "no_header_band",
                                 info["header_conf"]))
            print(f"[CROP] {page_id}: header={info['header_end_y']}"
                  f"({info['header_conf']}) split={info['split_x']}"
                  f"({info['gutter_conf']}) L={info['colL_w']} R={info['colR_w']}",
                  flush=True)
        except Exception as e:
            errored += 1
            failures.append((page_id, "page", "exception", str(e)))
            print(f"[CROP] {page_id}: ERROR {e}", file=sys.stderr, flush=True)

    # write failures (overwrite; full current view)
    with open(fail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "stage", "reason", "detail"])
        for row in sorted(set(failures)):
            w.writerow(row)

    print(f"[CROP] DONE: total={len(images)} ok={done} "
          f"(processed={processed} skipped={skipped}) errored={errored} "
          f"flagged={len(failures)} -> {fail_path}", flush=True)


if __name__ == "__main__":
    main()
