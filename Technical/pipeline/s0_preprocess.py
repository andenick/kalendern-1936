#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalendern reading pipeline — Stage 0: preprocessing.

CPU-ONLY. Never touches the GPU.

For each input page (a ~3000px-tall, two-column 1936 Swedish tax-directory scan):
  1. Deskew  — estimate skew via the projection-profile variance method
               (robust for dense text columns), with a Hough-line cross-check;
               rotate to correct.
  2. Grayscale.
  3. Binarize — BOTH Otsu and Sauvola (adaptive) are produced.
  4. Despeckle — a light morphological open on the binary to drop salt noise.
  5. Quality probe — Laplacian variance (focus/contrast proxy) + mean ink
               density → a per-page score that flags faded / low-contrast pages
               for later gated super-resolution (NOT applied here).

Outputs (per page, into --outdir):
  <page>_gray.png          deskewed grayscale
  <page>_bin.png           deskewed binarized (chosen method, default sauvola)
  <page>_bin_otsu.png      Otsu binarization (for comparison)
  <page>_bin_sauvola.png   Sauvola binarization (for comparison)
and appends a row to <outdir>/s0_quality.json (one JSON object per page).

Designed to be imported (functions are reusable by s2_column_split) or run as a CLI.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # hard GPU lockout
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import json
import argparse

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Deskew
# --------------------------------------------------------------------------- #
def _projection_variance_score(binary_row_dark, angle, h, w):
    """Rotate the dark-pixel image by `angle` and return the variance of the
    horizontal projection profile. Text lines maximize this variance when the
    page is level (rows are either all-ink or all-white)."""
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(binary_row_dark, M, (w, h),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    proj = rot.sum(axis=1, dtype=np.float64)
    return proj.var()


def estimate_skew_projection(gray, angle_range=4.0, coarse=0.5, fine=0.1):
    """Estimate skew angle (degrees) by maximizing horizontal-projection
    variance over candidate rotations. Two-pass coarse→fine search.
    A downscaled binary is used for speed."""
    h0, w0 = gray.shape
    scale = 1000.0 / max(h0, w0)
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    # dark pixels = 1
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bw = bw.astype(np.float32)
    h, w = bw.shape

    def best_over(angles):
        scores = [(_projection_variance_score(bw, a, h, w), a) for a in angles]
        return max(scores, key=lambda t: t[0])

    coarse_angles = np.arange(-angle_range, angle_range + 1e-9, coarse)
    _, a_coarse = best_over(coarse_angles)
    fine_angles = np.arange(a_coarse - coarse, a_coarse + coarse + 1e-9, fine)
    _, a_fine = best_over(fine_angles)
    return float(a_fine)


def estimate_skew_hough(gray):
    """Cross-check skew via Hough lines on edges. Returns angle in degrees or
    None if no reliable estimate."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 360.0, threshold=300)
    if lines is None:
        return None
    angles = []
    for rho_theta in lines[:200]:
        theta = rho_theta[0][1]
        deg = np.degrees(theta) - 90.0   # relative to horizontal
        if -10.0 < deg < 10.0:
            angles.append(deg)
    if not angles:
        return None
    return float(np.median(angles))


def deskew(gray):
    """Return (deskewed_gray, angle_deg). Projection method is primary; Hough
    is a sanity cross-check (used only if it broadly agrees)."""
    a_proj = estimate_skew_projection(gray)
    a_hough = estimate_skew_hough(gray)
    angle = a_proj
    if a_hough is not None and abs(a_hough - a_proj) < 1.5:
        # gentle blend toward agreement
        angle = 0.5 * (a_proj + a_hough)
    if abs(angle) < 0.05:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return rot, float(angle)


# --------------------------------------------------------------------------- #
# Binarization
# --------------------------------------------------------------------------- #
def binarize_otsu(gray):
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw


def binarize_sauvola(gray, window=31, k=0.2, R=128.0):
    """Sauvola adaptive threshold. T = m * (1 + k*((s/R) - 1)).
    Implemented with integral-image mean/std (cv2 boxFilter)."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(window, window),
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(g * g, ddepth=-1, ksize=(window, window),
                            borderType=cv2.BORDER_REFLECT)
    var = np.clip(mean_sq - mean * mean, 0, None)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * ((std / R) - 1.0))
    bw = np.where(g > thresh, 255, 0).astype(np.uint8)
    return bw


def despeckle(binary, ksize=2):
    """Light morphological open to remove isolated specks. `binary` is
    text=black(0) on white(255); open on the ink (invert, open, re-invert)."""
    ink = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.bitwise_not(ink)


# --------------------------------------------------------------------------- #
# Quality probe
# --------------------------------------------------------------------------- #
def quality_probe(gray, binary):
    """Return a dict of quality metrics + a 0-100 score and a faded flag."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ink_density = float((binary == 0).mean())          # fraction dark
    contrast = float(gray.std())
    mean_lum = float(gray.mean())

    # Heuristic score, calibrated to THIS corpus (1936 SSA scans cluster at
    # lap_var ~100-135, contrast ~22-27, ink ~0.09-0.11 -- soft-focus but legible).
    # Normalizers are set so a healthy page lands ~70-85 and degraded pages drop.
    sharp_n = min(lap_var / 130.0, 1.0)
    contrast_n = min(contrast / 28.0, 1.0)
    ink_n = min(ink_density / 0.10, 1.0)               # ~10% ink = healthy
    score = round(100.0 * (0.45 * sharp_n + 0.35 * contrast_n + 0.20 * ink_n), 1)

    # faded/low-contrast flag fires for OUTLIERS below the corpus cohort, not
    # the whole (uniformly soft) corpus. A page is flagged for later gated
    # super-res only if it is materially worse than a typical SSA page.
    faded = (lap_var < 80.0) or (contrast < 19.0) or (ink_density < 0.055)
    return {
        "laplacian_var": round(lap_var, 1),
        "ink_density": round(ink_density, 4),
        "contrast_std": round(contrast, 1),
        "mean_luminance": round(mean_lum, 1),
        "quality_score": score,
        "faded_flag": bool(faded),
    }


# --------------------------------------------------------------------------- #
# Page driver
# --------------------------------------------------------------------------- #
def preprocess_page(img_path, outdir, page_id=None, bin_method="sauvola",
                    write=True):
    """Full Stage-0 on one page. Returns a dict with paths + quality + skew.
    Also returns in-memory arrays under keys 'gray','bin' for downstream reuse."""
    if page_id is None:
        page_id = os.path.splitext(os.path.basename(img_path))[0]

    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if color is None:
        raise IOError(f"cannot read image: {img_path}")
    gray0 = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    gray, angle = deskew(gray0)

    otsu = binarize_otsu(gray)
    sauv = binarize_sauvola(gray)
    sauv = despeckle(sauv)
    otsu = despeckle(otsu)
    chosen = sauv if bin_method == "sauvola" else otsu

    q = quality_probe(gray, chosen)
    q.update({"page": page_id, "skew_angle_deg": round(angle, 3),
              "bin_method": bin_method,
              "height": int(gray.shape[0]), "width": int(gray.shape[1])})

    paths = {}
    if write:
        os.makedirs(outdir, exist_ok=True)
        paths["gray"] = os.path.join(outdir, f"{page_id}_gray.png")
        paths["bin"] = os.path.join(outdir, f"{page_id}_bin.png")
        paths["bin_otsu"] = os.path.join(outdir, f"{page_id}_bin_otsu.png")
        paths["bin_sauvola"] = os.path.join(outdir, f"{page_id}_bin_sauvola.png")
        cv2.imwrite(paths["gray"], gray)
        cv2.imwrite(paths["bin"], chosen)
        cv2.imwrite(paths["bin_otsu"], otsu)
        cv2.imwrite(paths["bin_sauvola"], sauv)
    q["paths"] = paths

    return {"page": page_id, "quality": q, "gray": gray,
            "bin": chosen, "otsu": otsu, "sauvola": sauv, "angle": angle}


def _append_quality(outdir, q):
    jpath = os.path.join(outdir, "s0_quality.json")
    data = []
    if os.path.exists(jpath):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data = [d for d in data if d.get("page") != q.get("page")]
    data.append(q)
    data.sort(key=lambda d: d.get("page", ""))
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Kalendern Stage 0 preprocessing")
    ap.add_argument("images", nargs="+", help="input page image(s)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bin", default="sauvola", choices=["sauvola", "otsu"])
    args = ap.parse_args()

    for img in args.images:
        page_id = os.path.splitext(os.path.basename(img))[0]
        try:
            r = preprocess_page(img, args.outdir, page_id=page_id,
                                bin_method=args.bin)
            _append_quality(args.outdir, r["quality"])
            q = r["quality"]
            print(f"[S0] {page_id}: skew={q['skew_angle_deg']:+.2f} "
                  f"score={q['quality_score']} faded={q['faded_flag']} "
                  f"lapvar={q['laplacian_var']} ink={q['ink_density']}")
        except Exception as e:
            print(f"[S0] {page_id}: ERROR {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
