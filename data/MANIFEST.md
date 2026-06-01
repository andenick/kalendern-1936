# Data Manifest — Kalendern

This repository ships **no data** and **no model weights**. Place inputs under
`DATA_ROOT` and model files under `MODELS_DIR` (see the project README).

## Source documents

### 1912 / 1914 — GUB digitisations (`kind: ocr_pdf`)
- **What:** Swedish *Taxeringskalender* volumes digitised by Göteborgs
  universitetsbibliotek (GUB), distributed as PDFs with an embedded OCR text layer.
- **Source:** Göteborgs universitetsbibliotek (GUB) digital collections —
  https://www.ub.gu.se/  (search the *Taxeringskalender* digitisations).
- **Expected layout:**
  ```
  DATA_ROOT/Inputs/GUB0128142.pdf   # taxkal_1912 (751 pp)
  DATA_ROOT/Inputs/GUB0126654.pdf   # taxkal_1914 (1065 pp)
  ```

### 1936 — Stockholm (`kind: image_scans`)
- **What:** 417 page photographs of the 1936 Bonnier *Taxeringskalender* for Greater
  Stockholm (two columns per page, A–Ö by surname).
- **Source:** Stockholm Stadsarkiv (SSA) — https://stadsarkivet.stockholm/
  (the physical volume is also held by major Swedish research libraries).
- **Expected layout:**
  ```
  DATA_ROOT/Inputs/Images/SSA_0001.* ... SSA_0417.*
  ```

## Models (set `MODELS_DIR`)
The image pipeline can use local OCR/VLM models. None are redistributed here; download
from their official sources:
- OCR engines: Surya (`surya-ocr`), PaddleOCR, docTR, Tesseract, Kraken, Calamari, eynollah.
- GGUF VLMs served via llama.cpp (`LLAMA_SERVER`): e.g. dots.ocr, GLM-OCR, olmOCR-2,
  Qwen3-VL-8B, PaddleOCR-VL — obtain GGUF builds (e.g. from Hugging Face) and place them
  under `MODELS_DIR`.

## Notes
- The *Taxeringskalender* volumes named above are early/mid-20th-century Swedish public
  tax records; check the holding institution's terms for reuse of their scans.
- The pipeline runs fully offline; no network access is required at run time once inputs
  and models are present locally.
