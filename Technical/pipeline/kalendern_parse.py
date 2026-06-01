#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""kalendern_parse.py -- decode the comma-based tax-directory records from the
canonical dots.ocr column reads into a structured table (CPU-only, deterministic).

Implements Jakob's domain spec (Inputs/Reference/KALENDERN_JAKOB_SPEC.md):
  entry = Surname, First-initials, Title, Location(parish code | municipality),
          municipal_income - state_income
  - leading "-"  : ditto surname from the entry above (recursive)
  - leading "- -": same PERSON as above, taxed in a 2nd location (first name dittoed)
  - income "a-b" : a = municipal, b = state; either side may be empty
  - income may bump to the next physical line (printer ran out of space)
  - "hustru"     : wife; location = husband's (the entry above)
  - "A.-B."      : company -> flag

Input : Outputs/reads/dots.ocr/<page>_colL.txt , _colR.txt  (reading order: L then R)
Output: Outputs/Data/kalendern_records.csv  + a QA summary printed + QA json.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
import csv, json, re, unicodedata
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
READS = PROJ / "Outputs" / "reads" / "dots.ocr"
OUTD = PROJ / "Outputs" / "Data"
OUTD.mkdir(parents=True, exist_ok=True)

PARISHES = {"A", "Bk", "Bm", "E", "Ee", "G", "H", "Hö", "J", "Jh", "Kh", "Kl",
            "Kt", "Ma", "Mt", "N", "O", "S", "SG"}
DASHES = "–—―‒-"        # en/em/figure dashes + hyphen
DASH_RE = re.compile(f"[{DASHES}]")
# income field: optional number, a dash, optional number (either side may be blank)
NUM = r"\d[\d  ]*"
INCOME_RE = re.compile(rf"^\s*(?P<m>{NUM})?\s*[{DASHES}]\s*(?P<s>{NUM})?\s*$")
ONLY_NUM_RE = re.compile(rf"^\s*{NUM}\s*$")
LEAD_DASH_RE = re.compile(rf"^\s*([{DASHES}](?:\s*[{DASHES}])*)\s*")


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", " ")


def clean_num(s: str | None):
    if not s:
        return None
    d = re.sub(r"[^\d]", "", s)
    return int(d) if d else None


def merge_lines(lines: list[str]) -> list[str]:
    """Merge printer income-bump lines: a line whose last field is a trailing
    dash with no 2nd number, OR that ends in a comma, absorbs a following
    income-only / number-only line."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        while i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            cur_stripped = cur.rstrip()
            # (a) word-break hyphenation: a letter+hyphen at EOL joined to a
            #     lowercase continuation (e.g. "Fidei-" + "kommisskapital").
            if re.search(r"[A-Za-zÅÄÖåäö]-$", cur_stripped) and re.match(r"^[a-zåäö]", nxt):
                cur = cur_stripped[:-1] + nxt
                i += 1
                continue
            # (b) printer income bump: a trailing comma or dash absorbs a
            #     following bare income / number line.
            needs = (cur_stripped.endswith(",")
                     or bool(re.search(f"[{DASHES}]\\s*$", cur_stripped)))
            cont = bool(INCOME_RE.match(nxt) or ONLY_NUM_RE.match(nxt))
            if needs and cont:
                cur = cur_stripped + " " + nxt
                i += 1
                continue
            break
        out.append(cur)
        i += 1
    return out


def split_income(fields: list[str]):
    """Pop a trailing income field if present -> (municipal, state, rest_fields)."""
    if not fields:
        return None, None, fields
    last = fields[-1].strip()
    m = INCOME_RE.match(last)
    if m:
        return clean_num(m.group("m")), clean_num(m.group("s")), fields[:-1]
    if ONLY_NUM_RE.match(last):       # a single bare number = municipal only
        return clean_num(last), None, fields[:-1]
    return None, None, fields


def parse_column(text: str, page: str, region: str, state: dict, rows: list):
    raw_lines = [ln for ln in norm(text).splitlines() if ln.strip()]
    for ln in merge_lines(raw_lines):
        line = ln.strip()
        if not line:
            continue
        # leading dashes -> ditto surname / same-person
        md = LEAD_DASH_RE.match(line)
        n_dash = 0
        if md:
            n_dash = len(DASH_RE.findall(md.group(1)))
            body = line[md.end():]
        else:
            body = line
        fields = [f.strip() for f in body.split(",")]
        muni, stat, rest = split_income(fields)

        same_person = n_dash >= 2          # double emdash = same individual, 2nd loc
        ditto_surname = n_dash >= 1

        if ditto_surname:
            surname = state.get("surname", "")
            rest_fields = rest
        else:
            surname = rest[0] if rest else ""
            state["surname"] = surname
            rest_fields = rest[1:]

        # first name / title / location out of the remaining fields
        # an entry looks like an institution (no personal first name) when the
        # field after the surname is itself a parish code or the surname carries
        # institution markers -- then there is NO given-name field to consume.
        inst_markers = re.compile(r"stiftelse|fond|kassa|kapital|A\.?\s*[–—-]\s*B\.?|"
                                  r"bolag|förening|stipend", re.I)
        institutional = bool(rest_fields) and (
            rest_fields[0] in PARISHES or bool(inst_markers.search(surname)))
        if same_person:
            given = state.get("given", "")
            title = state.get("title", "")
            loc_fields = rest_fields
        elif institutional:
            given = ""
            state["given"] = ""
            loc_fields = rest_fields
        else:
            given = rest_fields[0] if rest_fields else ""
            state["given"] = given
            loc_fields = rest_fields[1:]

        location = ""
        title_parts = []
        is_wife = False
        if loc_fields:
            # location = the LAST non-empty field; everything before = title
            location = loc_fields[-1].strip()
            title_parts = [f for f in loc_fields[:-1] if f]
        title = ", ".join(title_parts) if not same_person else state.get("title", "")
        if title_parts:
            state["title"] = title
        if re.search(r"\bhustru\b", (title + " " + " ".join(loc_fields)), re.I):
            is_wife = True
        if is_wife:                         # wife shares husband's location
            location = state.get("location", location)
        elif location:
            state["location"] = location

        is_company = bool(re.search(r"A\.?\s*[–—-]\s*B\.?", line))
        parish_known = location in PARISHES

        rows.append({
            "page": page, "region": region,
            "surname": surname, "given_names": given,
            "title": title, "location": location,
            "municipal_income": muni, "state_income": stat,
            "is_continuation_surname": int(ditto_surname),
            "is_same_person_2nd_loc": int(same_person),
            "is_wife": int(is_wife), "is_company": int(is_company),
            "location_is_known_parish": int(parish_known),
            "raw": line,
        })


def main() -> int:
    pages = sorted({p.stem.rsplit("_", 1)[0] for p in READS.glob("SSA_*_col*.txt")})
    rows: list = []
    state: dict = {}
    for page in pages:
        for region in ("colL", "colR"):
            f = READS / f"{page}_{region}.txt"
            if f.exists():
                parse_column(f.read_text(encoding="utf-8", errors="replace"),
                             page, region, state, rows)
    # write CSV
    out = OUTD / "kalendern_records.csv"
    cols = ["record_id", "page", "region", "surname", "given_names", "title",
            "location", "municipal_income", "state_income",
            "is_continuation_surname", "is_same_person_2nd_loc", "is_wife",
            "is_company", "location_is_known_parish", "raw"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            r["record_id"] = f"K1936_{i+1:06d}"
            w.writerow(r)

    # QA
    n = len(rows)
    have_income = sum(1 for r in rows if r["municipal_income"] or r["state_income"])
    have_loc = sum(1 for r in rows if r["location"])
    known_parish = sum(r["location_is_known_parish"] for r in rows)
    companies = sum(r["is_company"] for r in rows)
    wives = sum(r["is_wife"] for r in rows)
    cont = sum(r["is_continuation_surname"] for r in rows)
    same_p = sum(r["is_same_person_2nd_loc"] for r in rows)
    # alphabetical monotonicity of NEW surnames (sanity of A-O ordering)
    new_surn = [r["surname"] for r in rows if not r["is_continuation_surname"] and r["surname"]]
    asc = sum(1 for a, b in zip(new_surn, new_surn[1:])
              if a.casefold() <= b.casefold())
    mono = round(asc / max(1, len(new_surn) - 1), 4)
    qa = {"records": n, "pct_with_income": round(have_income / max(1, n), 4),
          "pct_with_location": round(have_loc / max(1, n), 4),
          "known_parish_locations": known_parish, "companies_AB": companies,
          "wives": wives, "continuation_surnames": cont,
          "same_person_2nd_loc": same_p,
          "surname_alpha_monotonicity": mono, "pages": len(pages)}
    (OUTD / "kalendern_records_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[parse] " + json.dumps(qa, ensure_ascii=False))
    print(f"[parse] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
