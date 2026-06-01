#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parse_taxkal.py -- Phase 2: extract structured tax records from a GUB ocr_pdf volume.

Year-parameterized generalisation of pipeline/kalendern_parse.py for the multi-city,
OCR-dash-noisy 1912/1914 volumes. Consumes the coordinate-split, ad-filtered column body
from gub_columns.split_page; tracks ditto surnames, same-person 2nd locations, wives,
companies, income bump-down, and a best-effort city from the running header.

  python parse_taxkal.py --volume taxkal_1912
Outputs:
  Outputs/Data/kalendern_<year>_records.csv  + _QA.json
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, csv, glob, json, re, sys, unicodedata
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import gub_columns as gc  # noqa: E402

PROJ = _HERE.parent.parent
DASHES = "–—―‒-"
DASH_RE = re.compile(f"[{DASHES}]")
LEAD_DASH = re.compile(rf"^\s*([{DASHES}](?:\s*[{DASHES}])*)\s*")
NUMG = re.compile(r"\d[\d ]*\d|\d")
# income field = a trailing comma-field bearing >=1 number group (the last field)
KNOWN_CITIES = ["Stockholm", "Göteborg", "Falun", "Norrmalm", "Malmö", "Uppsala",
                "Norrköping", "Gävle", "Örebro", "Helsingborg", "Borås", "Eskilstuna",
                "Jönköping", "Karlstad", "Sundsvall", "Linköping", "Lund", "Halmstad",
                "Landskrona", "Kalmar", "Västerås", "Söderhamn", "Hudiksvall"]


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def clean_int(s: str):
    d = re.sub(r"[^\d]", "", s or "")
    return int(d) if d else None


def split_income(field: str):
    """Return (municipal, state) from a noisy income field. Handles dash/space/hyphen
    separators and one-sided values; uses dash position to assign single values."""
    f = field.strip()
    nums = NUMG.findall(f)
    nums = [n for n in nums if re.sub(r"\D", "", n)]
    if len(nums) >= 2:
        return clean_int(nums[0]), clean_int(nums[1])
    if len(nums) == 1:
        # one value: which side of a dash?
        m = DASH_RE.search(f)
        if m:
            return (clean_int(nums[0]), None) if f.index(nums[0]) < m.start() else (None, clean_int(nums[0]))
        return clean_int(nums[0]), None
    return None, None


def is_income_field(field: str) -> bool:
    return bool(NUMG.search(field)) and len(re.sub(r"\D", "", field)) >= 2


def merge_bumpdown(lines: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        while i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            needs = cur.endswith(",") or bool(re.search(f"[{DASHES}]\\s*$", cur)) or not is_income_field(cur.split(",")[-1])
            cont = bool(re.match(rf"^\s*[{DASHES}]?\s*\d", nxt)) and len(nxt) < 24
            if needs and cont:
                cur = cur.rstrip() + " " + nxt
                i += 1
            else:
                break
        out.append(cur)
        i += 1
    return out


def detect_city(rec: dict, current: str) -> str:
    """Best-effort city from the top spans (large font near page top)."""
    spans = rec.get("spans", [])
    if not spans:
        return current
    top = [s for s in spans if s["y0"] < rec["rect"][1] * 0.12]
    text = " ".join(s["t"] for s in sorted(top, key=lambda s: s["x0"]))
    for city in KNOWN_CITIES:
        if city.lower() in text.lower() or city[:5].lower() in text.lower().replace(" ", ""):
            return city
    return current


def parse_column(body: str, ctx: dict, rows: list, meta: dict):
    lines = merge_bumpdown([l for l in norm(body).splitlines() if l.strip()])
    for line in lines:
        s = line.strip()
        md = LEAD_DASH.match(s)
        n_dash = len(DASH_RE.findall(md.group(1))) if md else 0
        body_s = s[md.end():] if md else s
        fields = [f.strip() for f in body_s.split(",")]
        # income = trailing field if it looks like income
        muni = state = None
        if fields and is_income_field(fields[-1]):
            muni, state = split_income(fields[-1]); fields = fields[:-1]
        ditto = n_dash >= 1
        same_person = n_dash >= 2
        # a line whose first field is a TITLE (not a name) is a continuation of the
        # surname above (common for wives printed without a leading ditto dash).
        if not ditto and fields and re.fullmatch(
                r"(hustru|änkefru|enkefru|fru|fröken|fr\.|d:r|stärbh\.?)",
                fields[0].strip(), re.I):
            ditto = True
        if ditto:
            surname = ctx.get("surname", "")
            rest = fields
        else:
            surname = fields[0] if fields else ""
            ctx["surname"] = surname
            rest = fields[1:]
        if same_person:
            given = ctx.get("given", ""); loc_fields = rest
        else:
            given = rest[0] if rest else ""; ctx["given"] = given; loc_fields = rest[1:]
        location = ""
        title_parts = []
        if loc_fields:
            location = loc_fields[-1].strip().rstrip(".")
            title_parts = [f for f in loc_fields[:-1] if f]
        title = ", ".join(title_parts)
        is_wife = bool(re.search(r"\bhustru\b", body_s, re.I))
        if is_wife:
            location = ctx.get("location", location)
        elif location:
            ctx["location"] = location
        is_company = bool(re.search(r"A\.?\s*[–—-]\s*B\.?", s))
        if not (surname or given or muni or state):
            continue
        rows.append({**meta, "city": ctx.get("city", ""),
                     "surname": surname, "given_names": given, "title": title,
                     "location": location, "municipal_income": muni, "state_income": state,
                     "is_continuation_surname": int(ditto), "is_same_person_2nd_loc": int(same_person),
                     "is_wife": int(is_wife), "is_company": int(is_company), "raw": s})


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    args = ap.parse_args(argv)
    reg = json.loads((PROJ / "Technical/volumes.json").read_text(encoding="utf-8"))
    vol = next(v for v in reg["volumes"] if v["volume_id"] == args.volume)
    year = vol["year"]
    wdir = PROJ / f"Technical/gub/words_{year}"
    # body-page set from the quality CSV
    qcsv = PROJ / "Outputs" / "Reports" / f"gub_quality_{year}_pages.csv"
    body = {int(r["page"]) for r in csv.DictReader(qcsv.open(encoding="utf-8")) if r["is_body"] == "1"}

    rows = []
    ctx = {"city": ""}
    for f in sorted(glob.glob(str(wdir / "p*.json"))):
        p = int(Path(f).stem[1:])
        if p not in body:
            continue
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        ctx["city"] = detect_city(rec, ctx.get("city", ""))
        res = gc.split_page(rec)
        for col in ("colL", "colR"):
            parse_column(res["columns"][col]["body"], ctx, rows,
                         {"volume": args.volume, "year": year, "page": p, "region": col})

    outd = PROJ / "Outputs" / "Data"; outd.mkdir(parents=True, exist_ok=True)
    cols = ["record_id", "volume", "year", "city", "page", "region", "surname",
            "given_names", "title", "location", "municipal_income", "state_income",
            "is_continuation_surname", "is_same_person_2nd_loc", "is_wife", "is_company", "raw"]
    out = outd / f"kalendern_{year}_records.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            r["record_id"] = f"K{year}_{i+1:06d}"
            w.writerow({k: r.get(k, "") for k in cols})

    n = len(rows)
    inc = sum(1 for r in rows if r["municipal_income"] or r["state_income"])
    loc = sum(1 for r in rows if r["location"])
    cities = sorted({r["city"] for r in rows if r["city"]})
    new = [r["surname"] for r in rows if not r["is_continuation_surname"] and r["surname"]]
    asc = sum(1 for a, b in zip(new, new[1:]) if a.casefold() <= b.casefold())
    qa = {"volume": args.volume, "year": year, "records": n,
          "pct_with_income": round(inc / max(1, n), 4), "pct_with_location": round(loc / max(1, n), 4),
          "companies_AB": sum(r["is_company"] for r in rows), "wives": sum(r["is_wife"] for r in rows),
          "continuation": sum(r["is_continuation_surname"] for r in rows),
          "same_person_2nd_loc": sum(r["is_same_person_2nd_loc"] for r in rows),
          "surname_alpha_monotonicity": round(asc / max(1, len(new) - 1), 4),
          "cities_detected": cities, "body_pages": len(body)}
    (outd / f"kalendern_{year}_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False))
    print(f"[parse_taxkal] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
