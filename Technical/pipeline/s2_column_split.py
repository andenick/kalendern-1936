#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalendern reading pipeline — Stage 2: header + column split.

CPU-ONLY. Depends on Stage 0 output (deskewed gray + binarized).

Layout of a page:
  +-------------------------------------------------+
  |  page#   running header text   year             |  <- HEADER BAND
  +-------------------------------------------------+
  | col L text ...        |  col R text ...          |
  | col L text ...        |  col R text ...          |
  +-------------------------------------------------+
                          ^ central vertical gutter

Algorithm:
  1. HEADER BAND: take the horizontal ink-projection (dark px per row). The
     header is a short text band at the very top, separated from the body by a
     horizontal whitespace valley (and often a rule line). Find the first
     sustained low-ink valley below the top text → header ends there, body
     begins.
  2. CENTRAL GUTTER: on the BODY only, take the vertical ink-projection (dark
     px per column), smooth it, and search the MIDDLE THIRD for the minimum
     (the white gutter between the two columns) → split x.
  3. Crop header, left column, right column.

Outputs (per page, into --outdir):
  <page>_header.png   the top header band (from the gray image)
  <page>_colL.png     left column (gray)
  <page>_colR.png     right column (gray)
  (binarized variants written too with _bin suffix when --also-bin)

Also writes/updates <outdir>/s2_split.json with detection diagnostics.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import json
import argparse

import numpy as np
import cv2

# allow running standalone or imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_preprocess as s0


def _ink_profile_rows(binary):
    """dark pixels per row (text=0 on white=255)."""
    return (binary == 0).sum(axis=1).astype(np.float64)


def _ink_profile_cols(binary):
    """dark pixels per column."""
    return (binary == 0).sum(axis=0).astype(np.float64)


def _smooth(x, win):
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")


# --------------------------------------------------------------------------- #
# Header band detection
# --------------------------------------------------------------------------- #
def _find_text_bands(is_text, gap_tol):
    """Group consecutive (gap-tolerant) text rows into [start,end] bands."""
    bands = []
    n = len(is_text)
    i = 0
    while i < n:
        if not is_text[i]:
            i += 1
            continue
        start = i
        last = i
        j = i
        while j < n:
            if is_text[j]:
                last = j
            elif j - last > gap_tol:
                break
            j += 1
        bands.append((start, last))
        i = last + 1
    return bands


def detect_header(binary, max_frac=0.13, smooth_win=5):
    """Find the row where the header band ends and the body begins.

    Robust principle: the header (page number + running title + year, usually
    with a rule line) is one text band near the top, separated from the dense
    body by the SINGLE WIDEST whitespace valley in the top region. We:
      1. threshold rows into text/non-text;
      2. enumerate whitespace valleys (runs of non-text rows) in the top band,
         but ONLY those that come after the first real header text and before
         the body;
      3. pick the widest such valley -> the header/body separator;
      4. cut at the center of that valley (this sits just below any rule line,
         which is itself text/ink and thus bounds the valley above).

    This avoids fragile speck-vs-header classification: a thin book-edge speck
    at the very top is simply absorbed because the valley AFTER the header text
    is much wider than the gap after a speck only when the header is a single
    line -- so we additionally anchor "header text" as the LAST text row before
    the widest valley, and require the header to contain a substantive text row.

    Returns (header_end_y, confidence_str).
    """
    h, w = binary.shape
    rows = _smooth(_ink_profile_rows(binary), smooth_win)
    limit = int(h * max_frac)
    if limit < 20:
        return 0, "page_too_short"

    # Skip the very top margin zone: dark scanner/book edges live there and
    # mimic a wide text row. The header line always sits below this.
    margin_skip = max(int(0.02 * h), 24)

    seg = rows[:limit].copy()
    seg[:margin_skip] = 0.0                  # neutralize the top-edge band
    text_thr = max(0.012 * w, seg.max() * 0.10)
    is_text = seg > text_thr
    n = len(seg)

    # Anchor on the HEADER line, not a faint top-edge speck. The header is a
    # real text line, so it contains a row with substantial ink. Find the first
    # row whose ink is a meaningful fraction of the top-band peak; the band
    # around it is the header.
    strong_thr = max(0.04 * w, seg.max() * 0.30)
    strong = next((i for i in range(margin_skip, n) if seg[i] >= strong_thr), None)
    if strong is None:
        first_text = next((i for i in range(margin_skip, n) if is_text[i]), None)
        if first_text is None:
            return 0, "no_header_text"
        strong = first_text

    # The header is typically 1-2 lines (page#, title, year, rule). The
    # header/body separator is the deepest ink MINIMUM in a window just below
    # the header anchor -- it need not fall all the way to zero (body lines can
    # start close), so we use a local minimum rather than a zero-ink run.
    #
    # Define the header band as the contiguous strong text around `strong`
    # (tolerating the small intra-header gap between page# line and rule), then
    # search a bounded window below it for the minimum.
    # Use a TIGHT gap when growing the header band so we don't bridge across
    # the (sometimes small) header/body separator into the body. Also cap the
    # header band height: a real header is at most ~2-3 lines.
    gap_tol = max(int(0.0035 * h), 4)
    band_cap = strong + int(0.035 * h)       # ~130px max header band
    i = strong
    last_text = strong
    while i < n and i <= band_cap:
        if is_text[i]:
            last_text = i
        elif i - last_text > gap_tol:
            break
        i += 1
    band_end = last_text

    # search window for the separator minimum: from just past the header band
    # down by up to ~3% of page height (covers the rule + gap to first body line)
    win_lo = band_end + 1
    win_hi = min(band_end + int(0.035 * h), limit)
    if win_hi <= win_lo + 1:
        header_end = min(band_end + max(int(0.005 * h), 8), limit)
        return int(header_end), "weak_no_valley"

    window = seg[win_lo:win_hi]
    rel_min = int(np.argmin(window))
    header_end = win_lo + rel_min
    first_text = strong

    header_end = int(min(max(header_end, first_text + 2), limit))
    if header_end <= first_text + 2:
        return 0, "degenerate"
    return header_end, "ok"


# --------------------------------------------------------------------------- #
# Central gutter detection
# --------------------------------------------------------------------------- #
def detect_gutter(binary, body_top=0, mid_frac=0.34, smooth_win=21):
    """Find the central vertical gutter x-coordinate.

    Searches the middle `mid_frac` band of page width for the minimum of the
    smoothed vertical ink-projection (computed on the body region only).

    Returns (split_x, confidence_str, diag dict).
    """
    h, w = binary.shape
    body = binary[body_top:, :]
    cols = _smooth(_ink_profile_cols(body), smooth_win)

    lo = int(w * (0.5 - mid_frac / 2.0))
    hi = int(w * (0.5 + mid_frac / 2.0))
    window = cols[lo:hi]
    if window.size == 0:
        return w // 2, "empty_window", {}

    local_min_idx = int(np.argmin(window))
    split_x = lo + local_min_idx
    gutter_val = float(cols[split_x])

    # confidence: the gutter should be a pronounced minimum vs. the column
    # masses on either side. Compare to the median ink of the body columns.
    body_median = float(np.median(cols[cols > 0])) if np.any(cols > 0) else 1.0
    # mean ink in the column bodies (away from gutter & margins)
    left_mass = float(np.mean(cols[int(w*0.10):int(w*0.40)]))
    right_mass = float(np.mean(cols[int(w*0.60):int(w*0.90)]))
    ref = max((left_mass + right_mass) / 2.0, 1.0)
    ratio = gutter_val / ref   # low = clean gutter

    if ratio < 0.25:
        conf = "ok"
    elif ratio < 0.55:
        conf = "weak"
    else:
        conf = "poor"

    diag = {
        "search_lo": lo, "search_hi": hi,
        "gutter_ink": round(gutter_val, 1),
        "left_mass": round(left_mass, 1),
        "right_mass": round(right_mass, 1),
        "gutter_ratio": round(ratio, 3),
        "body_median": round(body_median, 1),
    }
    return int(split_x), conf, diag


# --------------------------------------------------------------------------- #
# Page driver
# --------------------------------------------------------------------------- #
def split_page(gray, binary, page_id, outdir, also_bin=True, write=True):
    """Run header + gutter detection on one preprocessed page; crop & save."""
    h, w = binary.shape

    header_end, hconf = detect_header(binary)
    split_x, gconf, gdiag = detect_gutter(binary, body_top=header_end)

    # crops from the gray image (primary OCR-feed)
    header_img = gray[:header_end, :] if header_end > 0 else None
    colL = gray[header_end:, :split_x]
    colR = gray[header_end:, split_x:]

    paths = {}
    if write:
        os.makedirs(outdir, exist_ok=True)
        if header_img is not None and header_img.shape[0] > 2:
            paths["header"] = os.path.join(outdir, f"{page_id}_header.png")
            cv2.imwrite(paths["header"], header_img)
        paths["colL"] = os.path.join(outdir, f"{page_id}_colL.png")
        paths["colR"] = os.path.join(outdir, f"{page_id}_colR.png")
        cv2.imwrite(paths["colL"], colL)
        cv2.imwrite(paths["colR"], colR)
        if also_bin:
            if header_end > 0:
                cv2.imwrite(os.path.join(outdir, f"{page_id}_header_bin.png"),
                            binary[:header_end, :])
            cv2.imwrite(os.path.join(outdir, f"{page_id}_colL_bin.png"),
                        binary[header_end:, :split_x])
            cv2.imwrite(os.path.join(outdir, f"{page_id}_colR_bin.png"),
                        binary[header_end:, split_x:])

    info = {
        "page": page_id,
        "page_h": int(h), "page_w": int(w),
        "header_end_y": int(header_end),
        "header_frac": round(header_end / h, 4),
        "header_conf": hconf,
        "split_x": int(split_x),
        "split_frac": round(split_x / w, 4),
        "gutter_conf": gconf,
        "colL_w": int(colL.shape[1]), "colR_w": int(colR.shape[1]),
        "colL_aspect": round(colL.shape[0] / max(colL.shape[1], 1), 3),
        "colR_aspect": round(colR.shape[0] / max(colR.shape[1], 1), 3),
        "gutter_diag": gdiag,
        "paths": paths,
    }
    return info


def _append_split(outdir, info):
    jpath = os.path.join(outdir, "s2_split.json")
    data = []
    if os.path.exists(jpath):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data = [d for d in data if d.get("page") != info.get("page")]
    data.append(info)
    data.sort(key=lambda d: d.get("page", ""))
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_image(img_path, outdir, bin_method="sauvola"):
    """End-to-end: Stage0 preprocess then Stage2 split, on a raw page image."""
    page_id = os.path.splitext(os.path.basename(img_path))[0]
    r = s0.preprocess_page(img_path, outdir, page_id=page_id,
                           bin_method=bin_method, write=True)
    s0._append_quality(outdir, r["quality"])
    info = split_page(r["gray"], r["bin"], page_id, outdir)
    _append_split(outdir, info)
    return info, r["quality"]


def main():
    ap = argparse.ArgumentParser(description="Kalendern Stage 2 column split")
    ap.add_argument("images", nargs="+", help="raw page image(s)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bin", default="sauvola", choices=["sauvola", "otsu"])
    args = ap.parse_args()

    for img in args.images:
        page_id = os.path.splitext(os.path.basename(img))[0]
        try:
            info, q = process_image(img, args.outdir, bin_method=args.bin)
            print(f"[S2] {page_id}: header_end={info['header_end_y']} "
                  f"({info['header_frac']:.1%},{info['header_conf']}) "
                  f"split_x={info['split_x']} ({info['split_frac']:.1%},"
                  f"{info['gutter_conf']}) "
                  f"L={info['colL_w']}px R={info['colR_w']}px "
                  f"ratio={info['gutter_diag'].get('gutter_ratio')}")
        except Exception as e:
            print(f"[S2] {page_id}: ERROR {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
