#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalendern reading pipeline -- Stage 3: gold-free multi-engine fusion.

CPU-ONLY. No GPU, no network, no gold reference.

Given the SAME region (e.g. one column crop) read independently by several OCR
engines (doctr, tesseract-swe, kraken+CATMuS, easyocr, ...), fuse them into one
best transcription with a PER-LINE confidence -- using ONLY the engines'
agreement as the quality signal. There is no ground truth; agreement IS the
signal (the Consensus-Entropy thesis, arXiv 2504.11101): a line where the
engines converge is trustworthy; a line where they scatter is suspect.

REUSES the Grace consensus machinery:
  - ``consensus_entropy.ce_consensus_cluster(cands, sim_thresh)`` -> per-line
    inverse-entropy-weighted consensus pick + agreement fraction + CEResult
    (carrying delta, the low-consensus escalation signal).

The hard part for column reads is ALIGNMENT: each engine emits its own list of
lines for the same column, and those lists are *approximately parallel* (same
physical lines, same order) but not identical -- engines split/merge/drop lines
differently. We align them line-by-line with a robust order-preserving fuzzy
matcher (an anchor-and-fill scheme over a similarity matrix, no gold needed),
then run CE consensus per aligned line.

Public API
----------
``fuse_region(engine_texts: dict[str, str], *, sim_thresh=0.62,
              consensus_sim=0.78) -> dict``
    Returns {fused_text, lines, per_line_confidence, mean_agreement, n_engines,
             engines, mean_delta}.

Handles 2..N engines. With 1 engine it passes the text through at confidence 1.0
(nothing to disagree with). With 0 engines it returns empty.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import sys
from typing import Dict, List, Optional, Sequence, Tuple

# --- consensus machinery --------------------------------------------------- #
# This stage fuses multiple OCR engine outputs via a consensus-entropy module
# (`consensus_entropy`). Provide its location on CONSENSUS_HARNESS (a directory
# containing consensus_entropy.py) or have it importable on PYTHONPATH.
_HARNESS = os.environ.get("CONSENSUS_HARNESS")
if _HARNESS and _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

from consensus_entropy import (  # noqa: E402
    ce_consensus_cluster,
    similarity,
    ned,
)


# --------------------------------------------------------------------------- #
# Line splitting / normalization
# --------------------------------------------------------------------------- #
def split_lines(text: str) -> List[str]:
    """Split an engine's region text into non-empty, stripped lines."""
    if not text:
        return []
    out = []
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = ln.strip()
        if s:
            out.append(s)
    return out


def _norm(s: str) -> str:
    """Light normalization for the alignment similarity only (the fused output
    keeps original casing/spelling -- we never normalize the emitted text)."""
    return " ".join((s or "").lower().split())


# --------------------------------------------------------------------------- #
# Robust order-preserving line alignment (no gold).
# --------------------------------------------------------------------------- #
def _best_match(
    line: str,
    pool: Sequence[str],
    used: set,
    lo: int,
    hi: int,
    min_sim: float,
) -> Tuple[Optional[int], float]:
    """Best unused index in pool[lo:hi] by normalized similarity to ``line``."""
    nline = _norm(line)
    best_i, best_s = None, 0.0
    for i in range(lo, hi):
        if i in used:
            continue
        s = similarity(nline, _norm(pool[i]))
        if s > best_s:
            best_i, best_s = i, s
    if best_i is not None and best_s >= min_sim:
        return best_i, best_s
    return None, best_s


def align_engine_lines(
    engine_lines: Dict[str, List[str]],
    sim_thresh: float = 0.62,
) -> List[Dict[str, Optional[str]]]:
    """Align the parallel per-engine line lists into a list of "slots".

    Each slot is a dict ``{engine_name: line_or_None}`` -- the engines' reads of
    one physical line. Because the lists are approximately parallel and ordered,
    we use the engine with the MOST lines as the spine and, for every other
    engine, greedily match its lines to spine slots with an order-preserving
    fuzzy search (a forward-only pointer over a bounded window, so matching is
    monotone and O(N*W)). Unmatched lines from a non-spine engine are inserted as
    their own new slots at the right ordinal position, so no engine's content is
    silently dropped.

    No gold is used anywhere; alignment is purely engine-vs-engine similarity.
    """
    names = [n for n, ls in engine_lines.items() if ls]
    if not names:
        return []
    if len(names) == 1:
        n = names[0]
        return [{n: ln} for ln in engine_lines[n]]

    # spine = engine with the most lines (richest segmentation)
    spine = max(names, key=lambda n: len(engine_lines[n]))
    slots: List[Dict[str, Optional[str]]] = [
        {spine: ln} for ln in engine_lines[spine]
    ]

    for name in names:
        if name == spine:
            continue
        lines = engine_lines[name]
        used: set = set()
        ptr = 0  # forward-only pointer into slots (monotone alignment)
        window = 6  # how far ahead we let a line jump to find its slot
        # remember insertions so we can place leftovers in order
        leftovers: List[Tuple[int, str]] = []  # (slot_index_after, line)
        for li, line in enumerate(lines):
            lo = ptr
            hi = min(len(slots), ptr + window + 1)
            spine_pool = [slots[k].get(spine) or "" for k in range(len(slots))]
            j, s = _best_match(line, spine_pool, used, lo, hi, sim_thresh)
            if j is None:
                # also allow a small look-back (engine merged two spine lines)
                lo2 = max(0, ptr - 2)
                j, s = _best_match(line, spine_pool, used, lo2, lo, sim_thresh)
            if j is None:
                leftovers.append((ptr, line))
                continue
            slots[j][name] = line
            used.add(j)
            ptr = j + 1
        # splice leftovers in as their own slots, in order
        if leftovers:
            new_slots: List[Dict[str, Optional[str]]] = []
            li = 0
            for idx in range(len(slots) + 1):
                while li < len(leftovers) and leftovers[li][0] == idx:
                    new_slots.append({name: leftovers[li][1]})
                    li += 1
                if idx < len(slots):
                    new_slots.append(slots[idx])
            slots = new_slots
    return slots


# --------------------------------------------------------------------------- #
# Per-line CE consensus over the aligned slots.
# --------------------------------------------------------------------------- #
def fuse_region(
    engine_texts: Dict[str, str],
    *,
    sim_thresh: float = 0.62,
    consensus_sim: float = 0.78,
) -> Dict[str, object]:
    """Fuse N engines' reads of ONE region into a single gold-free transcription.

    Parameters
    ----------
    engine_texts   ``{engine_name: region_text}`` -- the same region as read by
                   each engine. 2..N engines expected; 1 or 0 handled gracefully.
    sim_thresh     line-ALIGNMENT similarity floor (looser): two engine lines are
                   "the same physical line" at/above this.
    consensus_sim  CE CLUSTER similarity (tighter): within an aligned line, two
                   reads count as the same transcription cluster at/above this
                   (passed straight to ``ce_consensus_cluster``).

    Returns a dict:
      fused_text          the joined consensus lines (original spelling kept).
      lines               list of per-line dicts {text, confidence, agreement,
                          delta, n_votes, votes:{engine:line}}.
      per_line_confidence list[float] aligned to ``lines`` (== agreement).
      mean_agreement      mean per-line agreement = the region quality signal.
      mean_delta          mean per-line CE delta (low-consensus / escalation).
      n_engines, engines  bookkeeping.
    """
    engines = [n for n, t in engine_texts.items() if (t and t.strip())]
    if not engines:
        return {
            "fused_text": "",
            "lines": [],
            "per_line_confidence": [],
            "mean_agreement": 0.0,
            "mean_delta": 0.0,
            "n_engines": 0,
            "engines": [],
        }

    engine_lines = {n: split_lines(engine_texts[n]) for n in engines}

    if len(engines) == 1:
        n = engines[0]
        lines = [
            {
                "text": ln,
                "confidence": 1.0,
                "agreement": 1.0,
                "delta": 0.0,
                "n_votes": 1,
                "votes": {n: ln},
            }
            for ln in engine_lines[n]
        ]
        return {
            "fused_text": "\n".join(l["text"] for l in lines),
            "lines": lines,
            "per_line_confidence": [1.0] * len(lines),
            "mean_agreement": 1.0 if lines else 0.0,
            "mean_delta": 0.0,
            "n_engines": 1,
            "engines": engines,
        }

    slots = align_engine_lines(engine_lines, sim_thresh=sim_thresh)

    out_lines: List[Dict[str, object]] = []
    confidences: List[float] = []
    deltas: List[float] = []
    for slot in slots:
        votes = {e: slot.get(e) for e in engines if slot.get(e)}
        cands = [v for v in votes.values() if v and v.strip()]
        if not cands:
            continue
        # CE consensus over this line's candidate reads (inverse-entropy weighted,
        # similarity-clustered -- the right read for short free-text lines).
        rep, agreement, ce = ce_consensus_cluster(cands, sim_thresh=consensus_sim)
        conf = float(agreement)  # agreement == 1 - (cluster disagreement)
        out_lines.append(
            {
                "text": rep,
                "confidence": round(conf, 4),
                "agreement": round(float(agreement), 4),
                "delta": round(float(ce.delta), 4),
                "n_votes": len(cands),
                "votes": votes,
            }
        )
        confidences.append(conf)
        deltas.append(float(ce.delta))

    mean_agr = sum(confidences) / len(confidences) if confidences else 0.0
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "fused_text": "\n".join(str(l["text"]) for l in out_lines),
        "lines": out_lines,
        "per_line_confidence": [round(c, 4) for c in confidences],
        "mean_agreement": round(mean_agr, 4),
        "mean_delta": round(mean_delta, 4),
        "n_engines": len(engines),
        "engines": engines,
    }


# --------------------------------------------------------------------------- #
# Convenience: fuse from a dir of per-engine read files for one region.
# --------------------------------------------------------------------------- #
def load_engine_reads(
    reads_root: str,
    page_id: str,
    region: str,
    engines: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Collect ``reads_root/<engine>/<page_id>_<region>.txt`` for each engine."""
    import glob

    texts: Dict[str, str] = {}
    if engines is None:
        engines = [
            d
            for d in os.listdir(reads_root)
            if os.path.isdir(os.path.join(reads_root, d))
        ]
    for eng in engines:
        path = os.path.join(reads_root, eng, f"{page_id}_{region}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                texts[eng] = f.read()
        else:
            # also accept the doctr-style flat name <page>_doctr.txt in regions/
            alts = glob.glob(os.path.join(reads_root, f"{page_id}_{eng}*.txt"))
            if alts:
                with open(alts[0], "r", encoding="utf-8", errors="replace") as f:
                    texts[eng] = f.read()
    return texts


# --------------------------------------------------------------------------- #
# Self-test / smoke
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    # Three "engines" reading the same 4-line Swedish column, with the kinds of
    # disagreement real OCR produces (a confident-majority line, a scattered
    # line, a dropped line, a split line).
    e1 = (
        "Andersson, Karl, direktor, 12 500\n"
        "Bergstrom, Anna, lararinna, 4 200\n"
        "Carlsson, Erik, ingenjor, 8 100\n"
        "Dahlgren, Olof, kopman, 15 000"
    )
    e2 = (
        "Andersson, Karl, direktor, 12 500\n"
        "Bergström, Anna, lärarinna, 4 200\n"
        "Carlsson, Erik, ingenjör, 8 100\n"
        "Dahlgren, Olof, köpman, 15 000"
    )
    e3 = (
        "Andersson, Karl, direktör, 12 500\n"
        "Bergstrorn, Aooa, lararinna, 4 ZOO\n"   # noisy
        "Carlsson, Erik, ingenjor, 8 100\n"
        "Dahlgren, Olof, kopman, 15 000"
    )
    res = fuse_region({"doctr": e1, "kblab": e2, "easyocr": e3})
    print("=== s3_fuse smoke (3 engines, gold-free) ===")
    print("engines:", res["engines"], "| mean_agreement:", res["mean_agreement"],
          "| mean_delta:", res["mean_delta"])
    for l in res["lines"]:
        print(f"  conf={l['confidence']:.2f} d={l['delta']:.2f} "
              f"n={l['n_votes']}  {l['text']}")
    print("--- fused_text ---")
    print(res["fused_text"])
    # 2-engine + 1-engine + 0-engine degenerate paths
    print("\n2-engine mean_agreement:",
          fuse_region({"a": e1, "b": e2})["mean_agreement"])
    print("1-engine mean_agreement:",
          fuse_region({"a": e1})["mean_agreement"])
    print("0-engine:", fuse_region({})["n_engines"])
    assert res["n_engines"] == 3
    assert 0.0 <= res["mean_agreement"] <= 1.0
    print("SMOKE_OK")


if __name__ == "__main__":
    _smoke()
