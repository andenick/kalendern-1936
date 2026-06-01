# Kalendern 1936 — Stockholm Tax Directory Extraction

Pipeline code to structure-extract Swedish *Taxeringskalender* (tax directories) into
per-person tax records. The flagship target is the 1936 Bonnier *Taxeringskalender* for
Greater Stockholm (high-income taxpayers, A–Ö by surname, two columns per page); the same
year-agnostic runner also handles earlier GUB-digitised volumes (1912, 1914) that ship with
an embedded OCR text layer.

This is a **code-only** release. The ~18.6 GB of page scans and OCR model weights are
**not** included — see [`data/MANIFEST.md`](data/MANIFEST.md) for sources.

## What it does

Two volume "kinds" (see `Technical/volumes.json`):

- **`ocr_pdf`** (GUB digitisations, e.g. 1912/1914): mine the embedded OCR text layer →
  word extraction → page classification → record parsing. CPU-only.
- **`image_scans`** (e.g. 1936 SSA photographs): an image OCR pipeline —
  preprocess → gutter/column split → multi-engine OCR (and/or a GGUF VLM such as dots.ocr
  served via llama.cpp) → consensus fusion → semantic parsing → A–Ö / income QA.

```
gutter-split -> per-column OCR [multi-engine + optional VLM] -> consensus fusion
            -> Python semantic rules (emdash / Hustru / parish / A.-B.) -> A-Ö & income QA
```

## Repository layout

- `Technical/run_volume.py` — year-agnostic entry point (`--list`, `--volume <id>`)
- `Technical/volumes.json` — volume registry
- `Technical/pipeline/` — image pipeline stages (`s0`–`s6`, assemble, parse, agreement report)
- `Technical/gub/` — GUB PDF-text path (extract / quality / parse) + dataset/LoRA tooling
- `Technical/ocr_engines/` — per-engine runners (PaddleOCR, docTR, Tesseract, Kraken, …)
- `Technical/vlm/` — llama.cpp GGUF VLM serving + prompts (`serve_model.ps1`)
- `Technical/run_surya.py` — Surya layout/recognition smoke run

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# then install whichever OCR engines you intend to run (see requirements.txt comments)
```

### Environment variables

Inputs are read from `DATA_ROOT`, outputs written to `OUTPUT_ROOT` (defaults `./data`,
`./outputs`). Additional optional vars point at external tools/assets:

| Variable | Purpose | Default |
|---|---|---|
| `DATA_ROOT` | source scans / PDFs (`Inputs/...`) | `data` |
| `OUTPUT_ROOT` | pipeline outputs | `outputs` |
| `NATIVE_PY` | Python interpreter for CPU OCR steps | current interpreter |
| `CONSENSUS_HARNESS` | dir containing `consensus_entropy.py` (multi-engine fusion); a dependency-free fallback is embedded | unset → fallback |
| `LLAMA_SERVER` | path to a llama.cpp `llama-server` build (VLM serving) | `llama-server.exe` on PATH |
| `MODELS_DIR` | dir holding the GGUF OCR/VLM model files | `models` |
| `GOLD_DIR` | human-verified gold pages (training/eval) | `data/gold` |
| `MODEL_DIR` | base model for LoRA fine-tuning | `models/LightOnOCR-2-1B` |
| `EVAL_HARNESS` | dir containing eval `scorers` (optional) | unset |

Copy `.env.example` to `.env` and edit if you prefer a dotenv file.

### Run

```bash
python Technical/run_volume.py --list
python Technical/run_volume.py --volume taxkal_1912
```

## API keys — bring your own

The pipeline runs **locally/offline** and needs **no API keys**. Heavy OCR/VLM models are
downloaded by their respective libraries (e.g. Hugging Face / PaddleOCR) on first use; if a
model is gated, set that library's standard auth (e.g. `HF_TOKEN`) per its own docs.

## Data

The source images (Stockholm Stadsarkiv photographs) and GUB digitisations are described in
[`data/MANIFEST.md`](data/MANIFEST.md). No data ships in this repository.

## License

No license is granted (all rights reserved) unless one is added later.
