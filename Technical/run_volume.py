#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_volume.py -- Phase 5: one year-agnostic entry point for any Kalendern volume.

Reads Technical/volumes.json and runs the right pipeline by `kind`:
  kind == "ocr_pdf"      (GUB digitisations: 1912, 1914, future):
      gub_extract (words) -> gub_quality (page classification) -> parse_taxkal (records)
      => Outputs/Data/kalendern_<year>_records.csv + _QA.json   [CPU only]
  kind == "image_scans"  (e.g. 1936 SSA photos):
      the image OCR pipeline (s0 -> s2 -> dots.ocr [GPU, single-launch] ->
      run_assemble_all -> kalendern_parse). The dots.ocr step needs the GPU and is
      run via Technical/vlm/serve_model.ps1 + race_e2e; this orchestrator prints the
      exact steps rather than seizing the GPU implicitly.

  python run_volume.py --volume taxkal_1912
  python run_volume.py --list
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REG = HERE / "volumes.json"
# Python interpreter used for the CPU OCR steps. Defaults to the current
# interpreter; override with NATIVE_PY to point at a dedicated (e.g. GPU) venv.
NATIVE_PY = os.environ.get("NATIVE_PY", sys.executable)
GUB = HERE / "gub"


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_ocr_pdf(vol):
    vid = vol["volume_id"]
    print(f"[run_volume] {vid} ({vol['year']}) -- ocr_pdf path (CPU)")
    run([NATIVE_PY, str(GUB / "gub_extract.py"), "--volume", vid, "--no-images"])
    run([NATIVE_PY, str(GUB / "gub_quality.py"), "--volume", vid])
    run([NATIVE_PY, str(GUB / "parse_taxkal.py"), "--volume", vid])
    print(f"[run_volume] done -> Outputs/Data/kalendern_{vol['year']}_records.csv")


def run_image_scans(vol):
    year = vol["year"]
    print(f"[run_volume] {vol['volume_id']} ({year}) -- image_scans path")
    print("  This path needs the GPU (dots.ocr) and runs serially single-launch.")
    print("  Steps (run from Technical/):")
    print(f"   1. pipeline/run_crop_all.py            # s0+s2 over {vol.get('image_dir','Inputs/Images')}")
    print( "   2. vlm/serve_model.ps1 -Model dots.ocr ; race_e2e over the column crops ; reap.ps1")
    print( "   3. pipeline/run_tess_all.py            # corroborator (optional)")
    print( "   4. pipeline/run_assemble_all.py        # readable + page_index + flagged")
    print(f"   5. pipeline/kalendern_parse.py         # -> Outputs/Data/kalendern_{year}_records.csv")
    print("  (1936 is already processed; see Outputs/Data/kalendern_records.csv.)")


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    reg = json.loads(REG.read_text(encoding="utf-8"))
    if args.list or not args.volume:
        print("volumes:")
        for v in reg["volumes"]:
            print(f"  {v['volume_id']:16s} {v['year']}  kind={v['kind']:12s} "
                  f"pages={v.get('n_pages','?')}  status={v.get('status','?')}")
        return 0
    vol = next((v for v in reg["volumes"] if v["volume_id"] == args.volume), None)
    if not vol:
        raise SystemExit(f"unknown volume {args.volume}")
    if vol["kind"] == "ocr_pdf":
        run_ocr_pdf(vol)
    elif vol["kind"] == "image_scans":
        run_image_scans(vol)
    else:
        raise SystemExit(f"unknown kind {vol['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
