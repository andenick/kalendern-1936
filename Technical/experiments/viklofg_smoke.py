import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL = "viklofg/swedish-ocr-correction"
print("Loading model (CPU)...", flush=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL)
# Model card: tokenizer is google/byt5-small (character/byte-level)
tok = AutoTokenizer.from_pretrained("google/byt5-small")
model.eval()
print("Loaded. arch:", model.config.model_type,
      "| params(M): %.1f" % (sum(p.numel() for p in model.parameters())/1e6), flush=True)


def correct(text: str) -> str:
    # byt5 = byte-level; ~128 UTF-8 byte limit per call
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**enc, max_length=512, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)


# Deliberately noisy Swedish OCR strings (å/ä/ö corruptions, OCR confusions like rn->m, l->I, 0->o)
noisy = [
    "Den i HandelstidniDgens g&rdagsnnmmer omtalade hvalfisken",  # model-card style example
    "Stockholrns stad ar belagen vid Malaren",                     # Stockholms ... är belägen vid Mälaren
    "Han bgrjade arbeta pa fabriken ar 1923",                       # Han började arbeta på fabriken år 1923
    "Goteborg och Malmo ar stora stader i Sverige",                 # Göteborg och Malmö är stora städer i Sverige
]

print("\n=== viklofg/swedish-ocr-correction smoke (CPU) ===")
for s in noisy:
    fixed = correct(s)
    print(f"NOISY : {s}")
    print(f"FIXED : {fixed}")
    print("-")
print("SMOKE_OK")
