#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_tess_all.py -- resumable full-corpus Tesseract-swe voter (CPU-only).

Reuses s4_engines.run_tesseract over every colL/colR crop in Outputs/regions/.
Idempotent: skips a region whose reads/tesseract/<page>_<region>.txt already exists
non-empty, so it resumes cleanly if interrupted by the foreground time window.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
import sys, time, glob
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import s4_engines as s4  # noqa: E402

REGIONS = _HERE.parent.parent / "Outputs" / "regions"
OUT = _HERE.parent.parent / "Outputs" / "reads" / "tesseract"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    crops = sorted(glob.glob(str(REGIONS / "*_colL.png")) + glob.glob(str(REGIONS / "*_colR.png")))
    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for c in crops:
        stem = Path(c).stem  # SSA_0002_colL
        outp = OUT / f"{stem}.txt"
        if outp.exists() and outp.stat().st_size > 0:
            n_skip += 1
            continue
        try:
            text = s4.run_tesseract(c)
            outp.write_text(text or "", encoding="utf-8")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"  ERR {stem}: {type(e).__name__}: {str(e)[:120]}")
        if (n_ok + n_err) % 50 == 0 and (n_ok + n_err) > 0:
            print(f"  ... {n_ok} ok, {n_skip} skip, {n_err} err, {time.time()-t0:.0f}s", flush=True)
    print(f"[tess-all] ok={n_ok} skip={n_skip} err={n_err} total={len(crops)} wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
