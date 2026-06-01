#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gub_columns.py -- coordinate-based two-column split + line assembly + ad filtering
for GUB OCR-PDF Kalendern pages (Phase 1/2).

Consumes a per-page words JSON from gub_extract.py and produces, deterministically from
the per-word coordinates:
  - the central gutter x (valley of word-straddle counts in the middle band)
  - left/right column line lists (y-clustered, x-ordered) in correct reading order
  - per-line tags: entry | ditto | header | ad/banner | other
  - an ad-filtered body text per column

No model, no GPU. Pure geometry + light Swedish-entry grammar.
"""
from __future__ import annotations
import re, statistics as st
from typing import Optional

DASHES = "–—―‒-"
# an "entry" line: has a comma, an income-ish digit run (>=3 digits), reasonable alpha.
_DIGIT3 = re.compile(r"\d{3,}")
_INCOME = re.compile(rf"\d[\d ]*\s*[{DASHES}]\s*\d|\b\d{{3,}}\b")
_LEAD_DITTO = re.compile(rf"^\s*[{DASHES}]")
# obvious ad / non-entry tokens (display banners, phone, address furniture)
_AD_TOKENS = re.compile(
    r"\b(A\.?\s?T\.?|R\.?\s?T\.?|Riks ?tel|Allm ?tel|Telegrafadress|Telefon|"
    r"Rikstelef|gatan \d|torg \d|Kr\.|KRONOR|PRIS|FRÅN|FRITT|MODERNA|"
    r"Konfektion|Skrädderi|Bankir|Byrå|import|IMPORT|fabrik|Fabrik)\b")


def detect_gutter(words: list[dict], page_w: float) -> tuple[float, float]:
    """Return (split_x, confidence). Gutter = x in the middle band [0.30w,0.70w]
    minimizing the number of words that straddle it; deeper valley => higher conf."""
    if not words:
        return page_w / 2, 0.0
    lo, hi = 0.30 * page_w, 0.70 * page_w
    xs = [lo + (hi - lo) * k / 40 for k in range(41)]
    def straddle(x):
        return sum(1 for w in words if w["x0"] < x < w["x1"])
    counts = [(straddle(x), abs(x - page_w / 2), x) for x in xs]
    best = min(counts, key=lambda c: (c[0], c[1]))
    # confidence: how much lower the valley is than the median straddle
    med = st.median([c[0] for c in counts]) or 1
    conf = max(0.0, min(1.0, 1 - best[0] / med)) if med else 0.0
    return best[2], round(conf, 3)


def _cluster_lines(words: list[dict]) -> list[str]:
    """y-cluster words into lines; within a line order by x; join by space."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w["y0"], w["x0"]))
    heights = [w["y1"] - w["y0"] for w in ws if w["y1"] > w["y0"]]
    tol = (st.median(heights) * 0.6) if heights else 6.0
    lines, cur, cy = [], [], None
    for w in ws:
        if cy is None or abs(w["y0"] - cy) <= tol:
            cur.append(w)
            cy = w["y0"] if cy is None else (cy + w["y0"]) / 2
        else:
            lines.append(cur); cur = [w]; cy = w["y0"]
    if cur:
        lines.append(cur)
    return [" ".join(t["t"] for t in sorted(ln, key=lambda w: w["x0"])) for ln in lines]


def classify_line(line: str, max_size: float, body_size: float) -> str:
    s = line.strip()
    if not s:
        return "blank"
    if _LEAD_DITTO.match(s) and ("," in s or _DIGIT3.search(s)):
        return "ditto"
    big = body_size and max_size > 1.6 * body_size
    is_entry = ("," in s) and bool(_INCOME.search(s)) and bool(re.search(r"[A-Za-zÅÄÖåäö]", s))
    if is_entry and not big:
        return "entry"
    if big or _AD_TOKENS.search(s) or (len(s) <= 3 and not _DIGIT3.search(s)):
        return "ad"
    # a running header line: contains a city/section word or a bare page number
    return "other"


def split_page(rec: dict) -> dict:
    """rec = a words-json dict. Returns column lines + tags + ad-filtered body."""
    page_w = rec["rect"][0]
    words = rec["words"]
    spans = rec.get("spans", [])
    sizes = [sp["size"] for sp in spans if sp.get("size")]
    body_size = st.median(sizes) if sizes else 0.0
    split_x, conf = detect_gutter(words, page_w)
    L = [w for w in words if (w["x0"] + w["x1"]) / 2 < split_x]
    R = [w for w in words if (w["x0"] + w["x1"]) / 2 >= split_x]
    out = {"split_x": round(split_x, 1), "gutter_conf": conf,
           "body_size": round(body_size, 1), "columns": {}}
    # map a coarse per-line max font size by y-overlap with spans
    def line_max_size(line_words):
        if not spans or not line_words:
            return body_size
        y0 = min(w["y0"] for w in line_words); y1 = max(w["y1"] for w in line_words)
        ov = [sp["size"] for sp in spans if not (sp["y1"] < y0 or sp["y0"] > y1)]
        return max(ov) if ov else body_size
    for tag, col in (("colL", L), ("colR", R)):
        raw_lines_words = _group_words_for_size(col)
        lines = _cluster_lines(col)
        tags = [classify_line(ln, line_max_size(gw), body_size)
                for ln, gw in zip(lines, raw_lines_words)]
        body = "\n".join(ln for ln, tg in zip(lines, tags) if tg in ("entry", "ditto"))
        out["columns"][tag] = {"lines": lines, "tags": tags, "body": body,
                               "n_lines": len(lines),
                               "n_entry": sum(t in ("entry", "ditto") for t in tags),
                               "n_ad": sum(t == "ad" for t in tags)}
    return out


def _group_words_for_size(words: list[dict]) -> list[list[dict]]:
    """Same y-clustering as _cluster_lines but returns the word groups (for size lookup)."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w["y0"], w["x0"]))
    heights = [w["y1"] - w["y0"] for w in ws if w["y1"] > w["y0"]]
    tol = (st.median(heights) * 0.6) if heights else 6.0
    groups, cur, cy = [], [], None
    for w in ws:
        if cy is None or abs(w["y0"] - cy) <= tol:
            cur.append(w); cy = w["y0"] if cy is None else (cy + w["y0"]) / 2
        else:
            groups.append(cur); cur = [w]; cy = w["y0"]
    if cur:
        groups.append(cur)
    return groups


if __name__ == "__main__":
    import json, sys
    rec = json.load(open(sys.argv[1], encoding="utf-8"))
    res = split_page(rec)
    print(f"split_x={res['split_x']} conf={res['gutter_conf']} body_size={res['body_size']}")
    for c in ("colL", "colR"):
        cc = res["columns"][c]
        print(f"\n--- {c}: {cc['n_entry']} entry/ditto of {cc['n_lines']} lines, {cc['n_ad']} ad ---")
        for ln, tg in list(zip(cc["lines"], cc["tags"]))[:12]:
            print(f"  [{tg:5s}] {ln[:70]}")
