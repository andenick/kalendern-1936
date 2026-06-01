#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_taxkal_lora.py -- Phase 4: LoRA fine-tune LightOnOCR-2-1B on the GUB silver
(column-image -> text) pairs, eval CER on a held-out slice BEFORE vs AFTER. GO/NO-GO for
domain fine-tuning toward the GUB-quality bar. bf16, no quant, single GPU, foreground.

Reuses the load_lightonocr vision-weight remap + masked-label LoRA recipe from
eval_harness/train_lora_minirun.py, but reads pairs from gold/sv_taxkal_*_manifest.csv.

  .venv-native/Scripts/python train_taxkal_lora.py --train 60 --eval 12 --steps 150
"""
from __future__ import annotations
import argparse, csv, os, random, sys, traceback
from pathlib import Path

MODEL = Path(os.environ.get("MODEL_DIR", "models/LightOnOCR-2-1B"))
GOLD = Path(os.environ.get("GOLD_DIR", "data/gold"))
MANIFESTS = [GOLD / "sv_taxkal_1912_manifest.csv", GOLD / "sv_taxkal_1914_manifest.csv"]
INSTR = "Transcribe this column of the Swedish tax directory, one entry per line."
MAX_SIDE = 1600   # cap the tall column crop's long side to bound vision tokens
csv.field_size_limit(10 ** 8)


def load_lightonocr(model_dir, dtype):
    """Remap vision_encoder.*/vision_projection.* -> vision_tower.*/multi_modal_projector.*
    so LightOnOCR's REAL vision weights load (else 222 tensors are random-init). See
    eval_harness/spike_lora_train.py."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(str(model_dir))
    model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
    raw = load_file(str(Path(model_dir) / "model.safetensors"))
    remapped = {k.replace("model.vision_encoder.", "model.vision_tower.")
                 .replace("model.vision_projection.", "model.multi_modal_projector."): v
                for k, v in raw.items()}
    res = model.load_state_dict(remapped, strict=False)
    bad = [k for k in res.missing_keys if "lm_head" not in k]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"load incomplete: {len(bad)} missing, {len(res.unexpected_keys)} unexpected")
    return model.to(dtype)


def _cer(ref: str, hyp: str) -> float:
    import unicodedata
    r = unicodedata.normalize("NFC", " ".join((ref or "").split()))
    h = unicodedata.normalize("NFC", " ".join((hyp or "").split()))
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def load_pairs(limit_chars=1200):
    out = []
    for man in MANIFESTS:
        if not man.exists():
            continue
        for r in csv.DictReader(man.open(encoding="utf-8")):
            p = Path(r["image_path"])
            ref = (r.get("reference") or "").strip()
            if p.exists() and len(ref) >= 80:
                out.append((p, ref[:limit_chars]))
    return out


def _fit(image):
    from PIL import Image
    w, h = image.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        image = image.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return image


def main(argv) -> int:
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args(argv)

    random.seed(0)
    pairs = load_pairs()
    random.shuffle(pairs)
    ev = pairs[:args.eval]
    tr = pairs[args.eval:args.eval + args.train]
    print(f"pairs total={len(pairs)} train={len(tr)} eval={len(ev)}", flush=True)

    dev = "cuda"
    try:
        proc = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True,
                                             fix_mistral_regex=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
    model = load_lightonocr(MODEL, torch.bfloat16).to(dev)

    def prompt_for():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTR}]}]
        try:
            return proc.apply_chat_template(msgs, add_generation_prompt=True)
        except Exception:
            return "<s>[INST] " + INSTR + " [/INST]"

    def featurize(img_path, target):
        image = _fit(Image.open(img_path).convert("RGB"))
        prompt = prompt_for()
        full = prompt + target + (proc.tokenizer.eos_token or "")
        enc = proc(text=full, images=[image], return_tensors="pt").to(dev)
        if "pixel_values" in enc and enc["pixel_values"].is_floating_point():
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        Lp = proc(text=prompt, images=[image], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :Lp] = -100
        enc["labels"] = labels
        return enc

    @torch.no_grad()
    def evaluate(tag):
        model.eval()
        cers = []
        for img_path, target in ev:
            image = _fit(Image.open(img_path).convert("RGB"))
            enc = proc(text=prompt_for(), images=[image], return_tensors="pt").to(dev)
            if "pixel_values" in enc and enc["pixel_values"].is_floating_point():
                enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
            hyp = proc.tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            cers.append(_cer(target, hyp))
        m = sum(cers) / len(cers)
        print(f"  [{tag}] eval CER = {m:.4f} (n={len(cers)})", flush=True)
        return m

    print("=== eval BASE (no LoRA) ===", flush=True)
    base = evaluate("base")

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    print(f"=== train {args.steps} steps ===", flush=True)
    model.train()
    feats = [featurize(p, t) for p, t in tr]
    import itertools
    loop = itertools.cycle(feats)
    run = 0.0
    for step in range(args.steps):
        enc = next(loop)
        opt.zero_grad()
        loss = model(**enc).loss
        loss.backward()
        opt.step()
        run += loss.item()
        if (step + 1) % 25 == 0:
            print(f"  step {step+1}: loss(avg25)={run/25:.4f}", flush=True)
            run = 0.0

    print("=== eval AFTER LoRA ===", flush=True)
    post = evaluate("lora")
    d = post - base
    print(f"\nMINI-RUN: base CER {base:.4f} -> LoRA CER {post:.4f}  (Δ {d:+.4f})")
    print("VERDICT:", "LoRA HELPS — domain-FT worth scaling" if d < -0.01
          else "LoRA NEUTRAL/HURTS at this scale — keep GUB-text/dots.ocr distillation path")
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM {vram:.1f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
