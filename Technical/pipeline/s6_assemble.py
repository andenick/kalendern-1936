#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalendern reading pipeline -- Stage 6: per-page assembly + master rendering.

CPU-ONLY. Pure-python (json/csv); no model, no GPU, no network.

Takes the per-page region reads produced upstream -- {header, page_number, colL,
colR} each with its fused confidence (from Stage 3) and optionally LM-corrected
(Stage 4/5) -- and:

  1. emits ONE structured JSON record per page (``assemble_page``);
  2. renders a readable page block: header band, then left column, then right
     column, in physical reading order (``render_page``);
  3. concatenates many page blocks into a master readable markdown
     (``assemble_master`` -> ``kalendern_1936_readable.md``);
  4. writes the SEPARATE header/page-number index ``page_index.csv``
     (ssa_id, page_number, header) via ``write_page_index``.

A "page read" dict (the input unit) looks like::

    {
      "ssa_id": "SSA_0001",
      "header":      {"text": "...", "confidence": 0.91},
      "page_number": {"text": "12",  "confidence": 0.99},   # may be embedded in header
      "colL":        {"text": "...", "confidence": 0.84},
      "colR":        {"text": "...", "confidence": 0.86},
    }

Each region value may also be a bare string (confidence defaults to None).
``page_number`` is optional -- if absent we try to recover it from the header.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import csv
import json
import re
from typing import Dict, List, Optional, Union

RegionVal = Union[str, Dict[str, object], None]

_PAGENO_RE = re.compile(r"\b(\d{1,4})\b")


# --------------------------------------------------------------------------- #
# region-value normalization
# --------------------------------------------------------------------------- #
def _region(val: RegionVal) -> Dict[str, object]:
    """Normalize a region value to {text, confidence}."""
    if val is None:
        return {"text": "", "confidence": None}
    if isinstance(val, str):
        return {"text": val, "confidence": None}
    if isinstance(val, dict):
        return {
            "text": str(val.get("text", "") or ""),
            "confidence": val.get("confidence"),
        }
    return {"text": str(val), "confidence": None}


def _extract_page_number(header_text: str) -> Optional[str]:
    """Recover a page number from the header band text.

    Kalendern page headers carry the page number at one edge (e.g. "12  Stockholm
    ... 1936" or "... 1936  13"). We pick the first 1-3 digit number that is NOT
    the year 1936 / 1935 (the header also prints those), preferring a number at
    the very start or very end of the header line."""
    if not header_text:
        return None
    toks = header_text.split()
    cands: List[str] = []
    for t in toks:
        m = _PAGENO_RE.search(t)
        if m:
            num = m.group(1)
            if num in {"1936", "1935", "1934"}:
                continue
            if len(num) <= 4:
                cands.append(num)
    if not cands:
        return None
    # prefer edge numbers (first or last token that matched)
    return cands[0]


# --------------------------------------------------------------------------- #
# (1) per-page JSON record
# --------------------------------------------------------------------------- #
def assemble_page(page_read: Dict[str, RegionVal]) -> Dict[str, object]:
    """Build the structured per-page JSON record from a page read dict."""
    ssa_id = str(page_read.get("ssa_id") or page_read.get("page") or "")
    header = _region(page_read.get("header"))
    colL = _region(page_read.get("colL"))
    colR = _region(page_read.get("colR"))

    pageno = page_read.get("page_number")
    if pageno is not None:
        pageno_r = _region(pageno)
    else:
        recovered = _extract_page_number(str(header["text"]))
        pageno_r = {"text": recovered or "", "confidence": None,
                    "source": "recovered_from_header" if recovered else "none"}

    confs = [r["confidence"] for r in (header, colL, colR)
             if isinstance(r.get("confidence"), (int, float))]
    page_conf = round(sum(confs) / len(confs), 4) if confs else None

    return {
        "ssa_id": ssa_id,
        "page_number": pageno_r,
        "header": header,
        "columns": {"colL": colL, "colR": colR},
        "page_confidence": page_conf,
        "n_chars": len(colL["text"]) + len(colR["text"]),
    }


# --------------------------------------------------------------------------- #
# (2) readable per-page rendering
# --------------------------------------------------------------------------- #
def render_page(record: Dict[str, object], *, with_conf: bool = True) -> str:
    """Render a per-page record to a readable markdown block:
    header band, then left column, then right column (physical reading order)."""
    ssa = record.get("ssa_id", "")
    pno = ""
    pn = record.get("page_number")
    if isinstance(pn, dict):
        pno = str(pn.get("text") or "")
    header = record.get("header", {}) or {}
    cols = record.get("columns", {}) or {}
    colL = cols.get("colL", {}) or {}
    colR = cols.get("colR", {}) or {}

    title = f"## {ssa}"
    if pno:
        title += f"  -- p. {pno}"
    if with_conf and record.get("page_confidence") is not None:
        title += f"   _(conf {record['page_confidence']})_"

    def _conf_tag(r: Dict[str, object]) -> str:
        c = r.get("confidence")
        return f" _(conf {c})_" if (with_conf and isinstance(c, (int, float))) else ""

    parts: List[str] = [title, ""]
    htext = str(header.get("text") or "").strip()
    if htext:
        parts.append(f"**Header:**{_conf_tag(header)}")
        parts.append("")
        parts.append("> " + htext.replace("\n", "  \n> "))
        parts.append("")
    parts.append(f"**Left column:**{_conf_tag(colL)}")
    parts.append("")
    parts.append(str(colL.get("text") or "").rstrip())
    parts.append("")
    parts.append(f"**Right column:**{_conf_tag(colR)}")
    parts.append("")
    parts.append(str(colR.get("text") or "").rstrip())
    parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# (3) master concatenation
# --------------------------------------------------------------------------- #
def assemble_master(
    records: List[Dict[str, object]],
    *,
    title: str = "Kalendern 1936 — Readable Transcription",
    with_conf: bool = True,
) -> str:
    """Concatenate many page records into one master readable markdown."""
    records = sorted(records, key=lambda r: str(r.get("ssa_id", "")))
    lines: List[str] = [f"# {title}", ""]
    npages = len(records)
    confs = [r["page_confidence"] for r in records
             if isinstance(r.get("page_confidence"), (int, float))]
    mean_conf = round(sum(confs) / len(confs), 4) if confs else None
    lines.append(f"_Pages: {npages}"
                 + (f" · mean page confidence: {mean_conf}_" if mean_conf is not None else "_"))
    lines.append("")
    lines.append("---")
    lines.append("")
    for rec in records:
        lines.append(render_page(rec, with_conf=with_conf))
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def write_master(records: List[Dict[str, object]], out_path: str,
                 **kwargs) -> str:
    md = assemble_master(records, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path


# --------------------------------------------------------------------------- #
# (4) the separate header / page-number index
# --------------------------------------------------------------------------- #
def write_page_index(records: List[Dict[str, object]], out_path: str) -> str:
    """Write page_index.csv: ssa_id, page_number, header (the header/page# index,
    kept separate from the body so it can be QA'd / joined independently)."""
    records = sorted(records, key=lambda r: str(r.get("ssa_id", "")))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ssa_id", "page_number", "header",
                    "header_conf", "page_confidence"])
        for rec in records:
            pn = rec.get("page_number") or {}
            header = rec.get("header") or {}
            htext = " ".join(str(header.get("text") or "").split())
            w.writerow([
                rec.get("ssa_id", ""),
                (pn.get("text") if isinstance(pn, dict) else "") or "",
                htext,
                header.get("confidence") if isinstance(header, dict) else "",
                rec.get("page_confidence", ""),
            ])
    return out_path


def write_page_json(record: Dict[str, object], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{record['ssa_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    import io, sys, tempfile

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    page_reads = [
        {
            "ssa_id": "SSA_0001",
            "header": {"text": "1936  Stockholm samt Stockholms lans stader  12",
                       "confidence": 0.88},
            "colL": {"text": "Andersson, Karl, direktör, 12 500\n"
                             "Bergström, Anna, lärarinna, 4 200",
                     "confidence": 0.84},
            "colR": {"text": "Carlsson, Erik, ingenjör, 8 100\n"
                             "Dahlgren, Olof, köpman, 15 000",
                     "confidence": 0.86},
        },
        {
            "ssa_id": "SSA_0002",
            "header": {"text": "13  Stockholm  1936", "confidence": 0.9},
            "page_number": {"text": "13", "confidence": 0.97},
            "colL": "Eriksson, Sven, snickare, 3 900",
            "colR": "Fransson, Maria, sömmerska, 2 800",
        },
    ]
    records = [assemble_page(p) for p in page_reads]
    print("=== s6 page record (SSA_0001) ===")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))
    print("\nrecovered page_number for SSA_0001:",
          records[0]["page_number"]["text"], "(expected 12)")
    assert records[0]["page_number"]["text"] == "12"
    assert records[1]["page_number"]["text"] == "13"

    with tempfile.TemporaryDirectory() as td:
        mpath = write_master(records, os.path.join(td, "kalendern_1936_readable.md"))
        ipath = write_page_index(records, os.path.join(td, "page_index.csv"))
        print("\n=== master md (head) ===")
        with open(mpath, encoding="utf-8") as f:
            print("".join(f.readlines()[:18]))
        print("=== page_index.csv ===")
        with open(ipath, encoding="utf-8") as f:
            print(f.read())
    print("SMOKE_OK")


if __name__ == "__main__":
    _smoke()
