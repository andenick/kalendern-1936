#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gold-free multi-engine OCR agreement / quality report for the Kalendern reads.

Goal
----
Maximize transcription accuracy *without gold* by using INTER-ENGINE AGREEMENT as
the confidence proxy. Where several independent OCR engines (paddle, tesseract*,
kraken, surya, and later dots.ocr / olmOCR) converge on the same characters, the
reading is almost certainly right; where they diverge, the page/column/line needs
review, post-correction, or future gold annotation.

This harness:
  1. Discovers every engine read for each (page, region).  Two on-disk layouts are
     supported and merged:
       * canonical   ``Outputs/reads/<engine>/<page>_<region>.txt``
       * actual      ``Outputs/regions/<page>_<engine>.txt``  (current Kalendern)
     Surya text additionally lives at ``Outputs/regions/<page>_surya_text.txt`` and
     carries ``===== [region N] ... =====`` separators which are stripped here.
  2. Computes, per (page, region):
       * a whole-region cross-engine agreement score (1 - CE delta, plus the
         CE mean off-diagonal similarity and a similarity-cluster consensus
         fraction) over the full normalized texts;
       * a per-line agreement via greedy best-match line alignment across engines
         (robust to the two-column interleave that breaks positional alignment),
         flagging HIGH-agreement lines (trust) vs LOW-agreement lines (review).
  3. Builds a pairwise inter-engine CER matrix (mean over shared regions): which
     engines agree most -> a gold-free proxy for which to trust / weight.
  4. Emits:
       * ``Outputs/AGREEMENT_REPORT.md``   - human-readable confidence map
       * ``Outputs/agreement_scores.csv``  - page,region,n_engines,mean_agreement,
                                             n_low_conf_lines (+ extra columns)
       * ``Outputs/agreement_lines.csv``   - per-line agreement (for triage / gold)
       * ``Outputs/agreement_pairwise_cer.csv`` - engine x engine CER matrix

It reuses a Consensus-Entropy module (``consensus_entropy.py``) when importable
on CONSENSUS_HARNESS / PYTHONPATH; otherwise it falls back to an embedded,
dependency-free copy of the same algorithm.  CPU-only, no GPU, no network:
pure-Python Levenshtein + (optional)
numpy.

Run
---
    set PYTHONUTF8=1
    set PYTHONIOENCODING=utf-8
    set CUDA_VISIBLE_DEVICES=
    python Technical\\pipeline\\agreement_report.py

Re-runnable: just run again as more engine reads land.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Paths.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # ...\Kalendern
OUTPUTS = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "Outputs"))
READS_DIR = OUTPUTS / "reads"                               # canonical layout
REGIONS_DIR = OUTPUTS / "regions"                           # actual layout
# Consensus-entropy harness (consensus_entropy.py). Set CONSENSUS_HARNESS to its
# directory, or have it importable on PYTHONPATH.
GRACE_CE = Path(os.environ.get("CONSENSUS_HARNESS", ""))

REPORT_MD = OUTPUTS / "AGREEMENT_REPORT.md"
SCORES_CSV = OUTPUTS / "agreement_scores.csv"
LINES_CSV = OUTPUTS / "agreement_lines.csv"
PAIRWISE_CSV = OUTPUTS / "agreement_pairwise_cer.csv"

# Agreement bands for flagging.
HIGH_AGREEMENT = 0.85      # >= this -> trusted line
LOW_AGREEMENT = 0.60       # <  this -> flagged for review / gold

# Engine filename token -> canonical engine name.  surya_text and tesseract_*
# variants are normalized; everything else passes through.
ENGINE_TOKEN_MAP = {
    "paddleocr": "paddle",
    "paddle": "paddle",
    "surya_text": "surya",
    "surya": "surya",
    "tesseract_swe": "tesseract_swe",
    "tesseract_frak": "tesseract_frak",
    "tesseract_frakswe": "tesseract_frakswe",
    "tesseract": "tesseract",
    "kraken": "kraken",
    "doctr": "doctr",
    "dots": "dots",
    "dots.ocr": "dots",
    "dotsocr": "dots",
    "olmocr": "olmocr",
    "olmocr-2-7b": "olmocr",
}

# Tokens that are NOT engine reads (layout / debug artifacts living in regions/).
NON_ENGINE_TOKENS = {"surya_layout", "quality", "split"}


# --------------------------------------------------------------------------- #
# Consensus-Entropy import (reuse Grace) with embedded fallback.
# --------------------------------------------------------------------------- #
def _levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein edit distance (no external deps)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def _try_import_grace_ce():
    """Import Grace's consensus_entropy; if its ``editdistance`` dep is missing,
    inject a shim so the module imports, then fall back to the embedded copy if
    even that fails."""
    if str(GRACE_CE) not in sys.path:
        sys.path.insert(0, str(GRACE_CE))
    try:
        import editdistance  # noqa: F401
    except Exception:
        # Provide a minimal shim module named ``editdistance`` exposing eval().
        import types
        shim = types.ModuleType("editdistance")
        shim.eval = _levenshtein  # type: ignore[attr-defined]
        sys.modules["editdistance"] = shim
    try:
        import consensus_entropy as _ce  # type: ignore
        return _ce
    except Exception:
        return None


_CE = _try_import_grace_ce()


# Embedded fallback (numpy-free) mirroring consensus_entropy.py semantics. -----
def _ned(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if not a and not b:
        return 0.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return _levenshtein(a, b) / denom


def _similarity(a: str, b: str) -> float:
    return 1.0 - _ned(a, b)


@dataclass
class CE:
    """Minimal CE result used by this harness (fallback shape)."""
    weights: List[float]
    entropies: List[float]
    delta: float
    mean_similarity: float
    n: int
    sim_matrix: List[List[float]]


def _row_entropy(probs: Sequence[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def _consensus_entropy_fallback(cands: Sequence[str], eps: float = 1e-9) -> CE:
    n = len(cands)
    if n == 0:
        return CE([], [], 0.0, 1.0, 0, [])
    if n == 1:
        return CE([1.0], [0.0], 0.0, 1.0, 1, [[1.0]])
    S = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _similarity(cands[i], cands[j])
            S[i][j] = S[j][i] = s
    entropies: List[float] = [0.0] * n
    nan_idx: List[int] = []
    for i in range(n):
        row = [S[i][k] for k in range(n) if k != i]
        rs = sum(row)
        if rs <= eps:
            nan_idx.append(i)
            continue
        entropies[i] = _row_entropy([v / rs for v in row])
    finite = [entropies[i] for i in range(n) if i not in nan_idx]
    max_ent = max(finite) if finite else math.log(n)
    for i in nan_idx:
        entropies[i] = max_ent
    inv = [1.0 / (e + eps) for e in entropies]
    tot = sum(inv)
    weights = [v / tot for v in inv] if tot > 0 else [1.0 / n] * n
    mean_e = sum(entropies) / n
    var = sum((e - mean_e) ** 2 for e in entropies) / n
    std = math.sqrt(var)
    norm = math.log(n) if n > 1 else 1.0
    delta = max(0.0, min(1.0, std / norm if norm > 0 else 0.0))
    off = [S[i][j] for i in range(n) for j in range(n) if i != j]
    mean_sim = sum(off) / len(off) if off else 1.0
    return CE(weights, entropies, delta, mean_sim, n, S)


def consensus_entropy(cands: Sequence[str]) -> CE:
    """Unified CE entrypoint: Grace module if available, else embedded fallback.

    Returns an object exposing .delta, .mean_similarity, .n, .sim_matrix, .weights.
    """
    if _CE is not None:
        r = _CE.consensus_entropy(cands)
        sm = r.sim_matrix.tolist() if hasattr(r.sim_matrix, "tolist") else r.sim_matrix
        return CE(
            weights=list(r.weights),
            entropies=list(r.entropies),
            delta=float(r.delta),
            mean_similarity=float(r.mean_similarity),
            n=int(r.n),
            sim_matrix=sm,
        )
    return _consensus_entropy_fallback(cands)


def cluster_consensus_fraction(cands: Sequence[str], sim_thresh: float = 0.80) -> float:
    """Fraction of CE trust mass that falls in the largest near-duplicate cluster.

    Single-linkage clustering over the CE similarity matrix at ``sim_thresh``; the
    returned value is the winning cluster's share of total weight -> a graded,
    informative agreement signal for long strings (full-region text)."""
    if _CE is not None:
        try:
            _, frac, _ = _CE.ce_consensus_cluster(cands, sim_thresh=sim_thresh)
            return float(frac)
        except Exception:
            pass
    ce = consensus_entropy(cands)
    n = ce.n
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if ce.sim_matrix[i][j] >= sim_thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    best = max(clusters.values(),
              key=lambda m: sum(ce.weights[i] for i in m))
    total = sum(ce.weights) or 1.0
    return sum(ce.weights[i] for i in best) / total


# --------------------------------------------------------------------------- #
# Text normalization.
# --------------------------------------------------------------------------- #
_SURYA_HEADER_RE = re.compile(r"^=====.*=====\s*$")
_MATH_TAG_RE = re.compile(r"</?math>")
_DASH_RE = re.compile(r"[‐‑‒–—―−]")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str, for_compare: bool = True) -> str:
    """Normalize an OCR string for cross-engine comparison.

    NFKC unicode-fold, unify the many dash glyphs to '-', drop surya <math> tags,
    collapse whitespace.  ``for_compare`` additionally casefolds (case differences
    between engines are not transcription errors we want to flag)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _MATH_TAG_RE.sub("", s)
    s = _DASH_RE.sub("-", s)
    s = _WS_RE.sub(" ", s).strip()
    if for_compare:
        s = s.casefold()
    return s


def strip_surya_headers(raw: str) -> str:
    """Remove ``===== [region N] ... =====`` separator lines from surya text."""
    return "\n".join(ln for ln in raw.splitlines()
                     if not _SURYA_HEADER_RE.match(ln))


def split_lines(raw: str, engine: str) -> List[str]:
    """Engine-aware line splitting; returns non-empty normalized-for-display lines."""
    if engine == "surya":
        raw = strip_surya_headers(raw)
    lines = []
    for ln in raw.splitlines():
        disp = _WS_RE.sub(" ", unicodedata.normalize("NFKC", ln)).strip()
        disp = _MATH_TAG_RE.sub("", disp).strip()
        if disp:
            lines.append(disp)
    return lines


# --------------------------------------------------------------------------- #
# Discovery.
# --------------------------------------------------------------------------- #
@dataclass
class ReadKey:
    page: str
    region: str   # "" for whole-page (no region split, current Kalendern case)

    def label(self) -> str:
        return self.page if not self.region else f"{self.page}/{self.region}"


def _canon_engine(token: str) -> str | None:
    t = token.lower()
    if t in NON_ENGINE_TOKENS:
        return None
    return ENGINE_TOKEN_MAP.get(t, t)


def discover_reads() -> Dict[Tuple[str, str], Dict[str, str]]:
    """Return {(page, region): {engine: raw_text}} merging both layouts.

    Canonical:  reads/<engine>/<page>_<region>.txt
    Actual:     regions/<page>_<engine>.txt   (region == "" whole page)
    """
    out: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)

    # --- canonical reads/<engine>/<page>_<region>.txt -------------------------
    if READS_DIR.is_dir():
        for eng_dir in sorted(READS_DIR.iterdir()):
            if not eng_dir.is_dir():
                continue
            engine = _canon_engine(eng_dir.name)
            if engine is None:
                continue
            for f in sorted(eng_dir.glob("*.txt")):
                stem = f.stem               # <page>_<region>
                m = re.match(r"^(.*?)_([A-Za-z]+\d*|col[LR]|r\d+|region\d+)$", stem)
                if m:
                    page, region = m.group(1), m.group(2)
                else:
                    page, region = stem, ""
                try:
                    raw = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                out[(page, region)][engine] = raw

    # --- actual regions/<page>_<engine>.txt -----------------------------------
    if REGIONS_DIR.is_dir():
        for f in sorted(REGIONS_DIR.glob("*.txt")):
            stem = f.stem
            m = re.match(r"^(SSA_\d+)_(.+)$", stem)
            if not m:
                continue
            page, token = m.group(1), m.group(2)
            engine = _canon_engine(token)
            if engine is None:
                continue
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # whole-page (no region split present in current Kalendern layout)
            out[(page, "")].setdefault(engine, raw)

    return out


# --------------------------------------------------------------------------- #
# Per-line greedy alignment + agreement.
# --------------------------------------------------------------------------- #
@dataclass
class LineGroup:
    rep: str                       # representative (longest) raw line
    members: Dict[str, str]        # engine -> raw line
    agreement: float               # 1 - CE delta over members' normalized text
    mean_sim: float                # CE mean off-diagonal similarity


def align_lines(engine_lines: Dict[str, List[str]],
                match_thresh: float = 0.55) -> List[LineGroup]:
    """Greedy best-match line alignment across engines (anchor = longest read).

    Positional alignment is unreliable here because some engines interleave the
    two columns line-by-line while others (surya) separate them.  Instead we anchor
    on the engine with the most lines and, for every other engine, greedily assign
    each of its lines to the best-matching unused anchor group (similarity >=
    ``match_thresh``); unmatched lines open new groups.  Each resulting group is a
    set of cross-engine lines believed to be the same physical line; its agreement
    is the CE consensus over the group's normalized members."""
    engines = list(engine_lines.keys())
    if not engines:
        return []
    anchor = max(engines, key=lambda e: len(engine_lines[e]))
    groups: List[Dict[str, str]] = [{anchor: ln} for ln in engine_lines[anchor]]
    group_norm: List[str] = [normalize_text(ln) for ln in engine_lines[anchor]]

    for eng in engines:
        if eng == anchor:
            continue
        used = [False] * len(groups)
        for ln in engine_lines[eng]:
            nln = normalize_text(ln)
            best_i, best_s = -1, match_thresh
            for i, gnorm in enumerate(group_norm):
                if used[i] or eng in groups[i]:
                    continue
                s = _similarity(nln, gnorm)
                if s > best_s:
                    best_i, best_s = i, s
            if best_i >= 0:
                groups[best_i][eng] = ln
                used[best_i] = True
            else:
                groups.append({eng: ln})
                group_norm.append(nln)
                used.append(True)

    out: List[LineGroup] = []
    for g in groups:
        members = g
        norm = [normalize_text(v) for v in members.values()]
        if len(norm) >= 2:
            ce = consensus_entropy(norm)
            agreement = 1.0 - ce.delta
            mean_sim = ce.mean_similarity
        else:
            # singleton line: only one engine saw it -> zero corroboration
            agreement = 0.0
            mean_sim = 0.0
        rep = max(members.values(), key=lambda s: len(s or ""))
        out.append(LineGroup(rep=rep, members=members,
                             agreement=agreement, mean_sim=mean_sim))
    return out


# --------------------------------------------------------------------------- #
# Region-level analysis.
# --------------------------------------------------------------------------- #
@dataclass
class RegionResult:
    page: str
    region: str
    engines: List[str]
    n_engines: int
    region_agreement: float        # 1 - CE delta over full normalized texts
    region_mean_sim: float         # CE mean off-diagonal similarity
    cluster_frac: float            # largest near-dup cluster trust fraction
    mean_line_agreement: float     # mean over multi-engine line groups
    n_lines: int                   # total line groups
    n_high_lines: int
    n_low_lines: int
    line_groups: List[LineGroup]


def analyze_region(page: str, region: str,
                   engine_raw: Dict[str, str]) -> RegionResult | None:
    engines = sorted(engine_raw.keys())
    if len(engines) < 2:
        return None

    # Whole-region consensus over full normalized texts.
    if region == "":  # surya raw has headers; strip per-engine inside split_lines
        full = {e: " ".join(split_lines(engine_raw[e], e)) for e in engines}
    else:
        full = {e: " ".join(split_lines(engine_raw[e], e)) for e in engines}
    full_norm = [normalize_text(full[e]) for e in engines]
    ce = consensus_entropy(full_norm)
    region_agreement = 1.0 - ce.delta
    region_mean_sim = ce.mean_similarity
    cluster_frac = cluster_consensus_fraction(full_norm)

    # Per-line alignment + agreement.
    engine_lines = {e: split_lines(engine_raw[e], e) for e in engines}
    groups = align_lines(engine_lines)
    multi = [g for g in groups if len(g.members) >= 2]
    mean_line_agreement = (sum(g.mean_sim for g in multi) / len(multi)
                           if multi else 0.0)
    n_high = sum(1 for g in multi if g.mean_sim >= HIGH_AGREEMENT)
    n_low = sum(1 for g in groups if g.mean_sim < LOW_AGREEMENT)

    return RegionResult(
        page=page, region=region, engines=engines, n_engines=len(engines),
        region_agreement=region_agreement, region_mean_sim=region_mean_sim,
        cluster_frac=cluster_frac, mean_line_agreement=mean_line_agreement,
        n_lines=len(groups), n_high_lines=n_high, n_low_lines=n_low,
        line_groups=groups,
    )


# --------------------------------------------------------------------------- #
# Pairwise inter-engine CER matrix.
# --------------------------------------------------------------------------- #
def pairwise_cer(results: List[RegionResult],
                 reads: Dict[Tuple[str, str], Dict[str, str]]) -> Tuple[
                     List[str], Dict[Tuple[str, str], float], Dict[Tuple[str, str], int]]:
    """Mean character error rate (NED) between each engine pair over shared regions.

    For each (page, region) where both engines produced a read, compute NED over the
    full normalized region text; average across all shared regions.  Lower = the two
    engines agree more (a gold-free proxy for clustering / trust)."""
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    engines_seen = set()

    for (page, region), engine_raw in reads.items():
        engs = sorted(engine_raw.keys())
        if len(engs) < 2:
            continue
        full_norm = {e: normalize_text(" ".join(split_lines(engine_raw[e], e)))
                     for e in engs}
        for e in engs:
            engines_seen.add(e)
        for i in range(len(engs)):
            for j in range(i + 1, len(engs)):
                a, b = engs[i], engs[j]
                d = _ned(full_norm[a], full_norm[b])
                sums[(a, b)] += d
                counts[(a, b)] += 1

    engines = sorted(engines_seen)
    matrix: Dict[Tuple[str, str], float] = {}
    nmat: Dict[Tuple[str, str], int] = {}
    for i, a in enumerate(engines):
        for j, b in enumerate(engines):
            if a == b:
                matrix[(a, b)] = 0.0
                nmat[(a, b)] = 0
                continue
            key = (a, b) if (a, b) in sums else (b, a)
            if key in sums and counts[key] > 0:
                matrix[(a, b)] = sums[key] / counts[key]
                nmat[(a, b)] = counts[key]
            else:
                matrix[(a, b)] = float("nan")
                nmat[(a, b)] = 0
    return engines, matrix, nmat


# --------------------------------------------------------------------------- #
# Output writers.
# --------------------------------------------------------------------------- #
def write_scores_csv(results: List[RegionResult]) -> None:
    with SCORES_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page", "region", "n_engines", "mean_agreement",
                    "n_low_conf_lines", "region_agreement", "region_mean_sim",
                    "cluster_consensus_frac", "mean_line_agreement",
                    "n_lines", "n_high_conf_lines", "engines"])
        for r in sorted(results, key=lambda x: (x.page, x.region)):
            w.writerow([
                r.page, r.region, r.n_engines,
                f"{r.region_mean_sim:.4f}",
                r.n_low_lines,
                f"{r.region_agreement:.4f}",
                f"{r.region_mean_sim:.4f}",
                f"{r.cluster_frac:.4f}",
                f"{r.mean_line_agreement:.4f}",
                r.n_lines, r.n_high_lines,
                "|".join(r.engines),
            ])


def write_lines_csv(results: List[RegionResult]) -> None:
    with LINES_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page", "region", "line_idx", "n_engines_on_line",
                    "agreement_mean_sim", "band", "representative", "engines"])
        for r in sorted(results, key=lambda x: (x.page, x.region)):
            for idx, g in enumerate(r.line_groups):
                if g.mean_sim >= HIGH_AGREEMENT:
                    band = "HIGH"
                elif g.mean_sim < LOW_AGREEMENT:
                    band = "LOW"
                else:
                    band = "MID"
                w.writerow([
                    r.page, r.region, idx, len(g.members),
                    f"{g.mean_sim:.4f}", band,
                    g.rep[:300], "|".join(sorted(g.members.keys())),
                ])


def write_pairwise_csv(engines: List[str],
                       matrix: Dict[Tuple[str, str], float],
                       nmat: Dict[Tuple[str, str], int]) -> None:
    with PAIRWISE_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["engine"] + engines)
        for a in engines:
            row = [a]
            for b in engines:
                v = matrix.get((a, b), float("nan"))
                row.append("" if v != v else f"{v:.4f}")
            w.writerow(row)
        w.writerow([])
        w.writerow(["# n_shared_regions matrix"])
        w.writerow(["engine"] + engines)
        for a in engines:
            w.writerow([a] + [str(nmat.get((a, b), 0)) for b in engines])


def _fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.3f}"


def write_report_md(results: List[RegionResult],
                    engines: List[str],
                    matrix: Dict[Tuple[str, str], float],
                    nmat: Dict[Tuple[str, str], int],
                    reads: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    res_sorted = sorted(results, key=lambda x: (x.page, x.region))
    n_regions = len(res_sorted)
    n_pairs_total = sum(len(v) for v in reads.values())
    multi = [r for r in res_sorted]
    mean_region_sim = (sum(r.region_mean_sim for r in multi) / len(multi)
                       if multi else 0.0)
    mean_cluster = (sum(r.cluster_frac for r in multi) / len(multi)
                    if multi else 0.0)
    total_lines = sum(r.n_lines for r in multi)
    total_high = sum(r.n_high_lines for r in multi)
    total_low = sum(r.n_low_lines for r in multi)
    used_grace = _CE is not None

    lines: List[str] = []
    A = lines.append
    A("# Kalendern Gold-Free Multi-Engine Agreement Report")
    A("")
    A("Inter-engine agreement as an accuracy proxy with **no gold**. Where "
      "independent OCR engines converge, the reading is trusted; where they "
      "diverge, the page/column/line is flagged for review, post-correction, or "
      "future gold annotation.")
    A("")
    A(f"- CE engine: **{'Grace race_harness consensus_entropy.py' if used_grace else 'embedded fallback (numpy-free)'}**")
    A(f"- Regions analysed (>=2 engines): **{n_regions}**")
    A(f"- Total engine reads on disk: **{n_pairs_total}**")
    A(f"- Mean cross-engine region similarity (1 - mean NED): **{mean_region_sim:.3f}**")
    A(f"- Mean cluster-consensus fraction (>=0.80 NED-cluster trust mass): **{mean_cluster:.3f}**")
    A(f"- Aligned line groups: **{total_lines}**  |  HIGH (>= {HIGH_AGREEMENT}): "
      f"**{total_high}**  |  LOW (< {LOW_AGREEMENT}): **{total_low}**")
    A("")
    A("Bands: **HIGH** = trust (>= {:.2f} mean line similarity), "
      "**LOW** = review (< {:.2f}), MID in between."
      .format(HIGH_AGREEMENT, LOW_AGREEMENT))
    A("")

    # Per-region table.
    A("## Per-region agreement (the gold-free confidence map)")
    A("")
    A("| page | region | n_eng | region_sim | cluster_frac | line_agree | "
      "lines | HIGH | LOW |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in res_sorted:
        reg = r.region or "(page)"
        A(f"| {r.page} | {reg} | {r.n_engines} | {r.region_mean_sim:.3f} | "
          f"{r.cluster_frac:.3f} | {r.mean_line_agreement:.3f} | "
          f"{r.n_lines} | {r.n_high_lines} | {r.n_low_lines} |")
    A("")

    # Pairwise CER matrix.
    A("## Pairwise inter-engine CER (mean NED over shared regions)")
    A("")
    A("Lower = the two engines agree more. The tightest-agreeing engines are the "
      "ones to trust / weight when fusing (gold-free clustering proxy).")
    A("")
    A("| engine | " + " | ".join(engines) + " |")
    A("|" + "---|" * (len(engines) + 1))
    for a in engines:
        cells = [_fmt(matrix.get((a, b), float("nan"))) for b in engines]
        A(f"| {a} | " + " | ".join(cells) + " |")
    A("")
    # Engine clustering summary: mean CER of each engine to all others.
    A("### Engine centrality (mean CER to all other engines)")
    A("")
    A("| engine | mean CER to others | rank |")
    A("|---|---|---|")
    centro = []
    for a in engines:
        vals = [matrix[(a, b)] for b in engines
                if a != b and matrix.get((a, b), float("nan")) == matrix.get((a, b))]
        if vals:
            centro.append((a, sum(vals) / len(vals)))
    centro.sort(key=lambda x: x[1])
    for rank, (a, v) in enumerate(centro, 1):
        A(f"| {a} | {v:.3f} | {rank} |")
    A("")
    if centro:
        A(f"Most central (lowest mean CER, the consensus anchor): "
          f"**{centro[0][0]}** ({centro[0][1]:.3f}). "
          f"Most divergent: **{centro[-1][0]}** ({centro[-1][1]:.3f}).")
        A("")

    # Lowest-agreement regions (attention list).
    A("## Lowest-agreement regions (need attention first)")
    A("")
    worst = sorted(res_sorted, key=lambda r: r.region_mean_sim)[:15]
    A("| page | region | region_sim | LOW lines / total |")
    A("|---|---|---|---|")
    for r in worst:
        reg = r.region or "(page)"
        A(f"| {r.page} | {reg} | {r.region_mean_sim:.3f} | "
          f"{r.n_low_lines}/{r.n_lines} |")
    A("")

    # Low-confidence breakdown: singletons (one engine only) vs genuine disagreement.
    n_singleton = sum(1 for r in res_sorted for g in r.line_groups
                      if len(g.members) < 2)
    n_disagree = sum(1 for r in res_sorted for g in r.line_groups
                     if len(g.members) >= 2 and g.mean_sim < LOW_AGREEMENT)
    A("## LOW-confidence lines: two kinds")
    A("")
    A(f"- **Singletons** (only one engine produced the line, no corroboration): "
      f"**{n_singleton}** -> these are alignment gaps / engine-specific reads; "
      "review or wait for more engines.")
    A(f"- **Genuine disagreement** (>=2 engines aligned but diverge, mean_sim < "
      f"{LOW_AGREEMENT}): **{n_disagree}** -> the real post-correction / gold "
      "candidates.")
    A("")
    A("### Sample disagreement lines (post-correction / gold candidates)")
    A("")
    shown = 0
    for r in res_sorted:
        low = [g for g in r.line_groups
               if g.mean_sim < LOW_AGREEMENT and len(g.members) >= 2]
        for g in low[:5]:
            variants = " || ".join(f"{e}: {v[:80]}"
                                   for e, v in sorted(g.members.items()))
            A(f"- `{r.page}` sim={g.mean_sim:.2f} -> {variants}")
            shown += 1
            if shown >= 25:
                break
        if shown >= 25:
            break
    if shown == 0:
        A("_(no multi-engine line fell below the LOW threshold)_")
    A("")
    A("### Sample HIGH-agreement lines (trusted, gold-free)")
    A("")
    hi_shown = 0
    for r in res_sorted:
        hi = [g for g in r.line_groups
              if g.mean_sim >= HIGH_AGREEMENT and len(g.members) >= 4]
        for g in hi[:8]:
            A(f"- `{r.page}` sim={g.mean_sim:.2f} ({len(g.members)} eng) -> "
              f"{g.rep[:90]}")
            hi_shown += 1
            if hi_shown >= 12:
                break
        if hi_shown >= 12:
            break
    A("")
    A("---")
    A("")
    A("Re-run `python Technical/pipeline/agreement_report.py` as more engine "
      "reads (dots.ocr, olmOCR, ...) land; the confidence map updates in place.")
    A("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    reads = discover_reads()
    if not reads:
        print("No engine reads found under "
              f"{READS_DIR} or {REGIONS_DIR}", file=sys.stderr)
        return 1

    results: List[RegionResult] = []
    for (page, region), engine_raw in sorted(reads.items()):
        rr = analyze_region(page, region, engine_raw)
        if rr is not None:
            results.append(rr)

    engines, matrix, nmat = pairwise_cer(results, reads)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_scores_csv(results)
    write_lines_csv(results)
    write_pairwise_csv(engines, matrix, nmat)
    write_report_md(results, engines, matrix, nmat, reads)

    if not args.quiet:
        n_multi = len(results)
        mean_sim = (sum(r.region_mean_sim for r in results) / n_multi
                    if n_multi else 0.0)
        print(f"[agreement] reads on disk: {sum(len(v) for v in reads.values())}")
        print(f"[agreement] regions with >=2 engines: {n_multi}")
        print(f"[agreement] mean cross-engine region similarity: {mean_sim:.4f}")
        print(f"[agreement] engines seen: {', '.join(engines)}")
        print(f"[agreement] CE source: "
              f"{'Grace consensus_entropy.py' if _CE is not None else 'embedded fallback'}")
        print(f"[agreement] wrote: {REPORT_MD}")
        print(f"[agreement] wrote: {SCORES_CSV}")
        print(f"[agreement] wrote: {LINES_CSV}")
        print(f"[agreement] wrote: {PAIRWISE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
