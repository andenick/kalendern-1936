#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""punct_scorer.py -- bake-off metric: how faithfully does a read preserve the
*structural* punctuation of the Kalendern entry format? Commas are the field
delimiter; the en/em-dash joins the two income figures. An OCR that drops commas
silently destroys the parser even at low CER, so we score these explicitly.

Metrics (per (reference, hypothesis), reference = GUB column text):
  comma_recall / comma_precision / comma_f1   -- ',' counts (the field delimiter)
  dash_recall                                 -- income en/em-dash retention
  digit_recall                                -- income digits retained (no number loss)
  field_count_ratio                           -- mean (#commas hyp / #commas ref) per line
  plus fair_cer passthrough (if available)    -- overall char accuracy

CPU-only. Wraps eval_harness/scorers.py:fair_cer when importable; else a pure CER.
"""
from __future__ import annotations
import os, re, sys, unicodedata
from pathlib import Path

_EVAL = os.environ.get("EVAL_HARNESS", "")
if _EVAL and _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)
try:
    from scorers import fair_cer as _fair_cer  # needs jiwer (in .venv-eval)
except Exception:
    _fair_cer = None

DASHES = "–—―‒-"
_DASH = re.compile(f"[{DASHES}]")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _count(s: str, ch: str) -> int:
    return s.count(ch)


def cer(ref: str, hyp: str) -> float:
    if _fair_cer is not None:
        try:
            r = _fair_cer(ref, hyp)
            return float(getattr(r, "cer", r))
        except Exception:
            pass
    r = " ".join(_norm(ref).split()); h = " ".join(_norm(hyp).split())
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def _safe_div(a, b):
    return a / b if b else (1.0 if a == 0 else 0.0)


def score(ref: str, hyp: str) -> dict:
    R, H = _norm(ref), _norm(hyp)
    rc, hc = _count(R, ","), _count(H, ",")
    matched_commas = min(rc, hc)
    rd = len(_DASH.findall(R)); hd = len(_DASH.findall(H))
    r_dig = sum(c.isdigit() for c in R); h_dig = sum(c.isdigit() for c in H)
    # per-line comma fidelity (field-count integrity)
    rl = [l for l in R.splitlines() if l.strip()]
    hl = [l for l in H.splitlines() if l.strip()]
    ratios = []
    for i, rline in enumerate(rl):
        rcl = _count(rline, ",")
        if rcl == 0:
            continue
        hcl = _count(hl[i], ",") if i < len(hl) else 0
        ratios.append(min(hcl, rcl) / rcl)
    return {
        "comma_recall": round(_safe_div(matched_commas, rc), 4),
        "comma_precision": round(_safe_div(matched_commas, hc), 4),
        "comma_f1": round(_safe_div(2 * matched_commas, rc + hc), 4),
        "dash_recall": round(_safe_div(min(rd, hd), rd), 4),
        "digit_recall": round(_safe_div(min(r_dig, h_dig), r_dig), 4),
        "field_count_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 1.0,
        "cer": round(cer(R, H), 4),
        "ref_commas": rc, "hyp_commas": hc,
    }


if __name__ == "__main__":
    # smoke: a read that drops commas should score low comma_recall even at modest CER
    ref = ("Adelsohn, E. G., hustru, 3860—4210\n"
           "Adler, N., dir., Turinge, 12490—11680")
    good = ref
    drops = ("Adelsohn E. G. hustru 3860—4210\n"
             "Adler N. dir. Turinge 12490 11680")  # commas + one dash gone
    print("good  :", score(ref, good))
    print("drops :", score(ref, drops))
