#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalendern reading pipeline -- Stage 5: Swedish post-OCR LM correction.

CPU-ONLY. No GPU, no network at run time (model is in the local HF cache).

Wraps **KBLab/swedish-ocr-correction** -- a byT5 (T5 arch, byte/character-level,
~0.3B) Swedish post-OCR corrector trained on Abbyy+Tesseract output over Swedish
newspapers 1818-2018. It is the production Stage-5 corrector for Kalendern:
strictly better than the viklofg sibling (CER 1.57 vs 1.92, WER 6.23 vs 7.41 on
the shared split) and Apache-2.0. See ``experiments\\SWEDISH_OCR_SURVEY.md``.

Model facts (verified from the downloaded config):
  - architecture: ``T5ForConditionalGeneration``
  - tokenizer:    ``ByT5Tokenizer`` (vocab 384, byte-level) -- ships in the repo,
                  so we load it straight from the model id (no google/byt5-small
                  needed, unlike viklofg).
  - input limit:  byte-level, ~128 UTF-8 bytes per call. å/ä/ö are 2 bytes each,
                  so a "line" must be fed in <=~110-char phrase chunks.

WHY span protection is mandatory for Kalendern
----------------------------------------------
This is a 1936 tax directory: the load-bearing content is SURNAMES, occupations,
addresses, and INCOME/TAX NUMBERS. A language-model corrector is trained to make
text look like fluent newspaper Swedish -- exactly the wrong instinct for a
NUMBER (it can transpose/normalize digits) or a NAME-INITIAL (it can drop the
period or merge it). So those data-bearing spans must survive verbatim.

Strategy (context-preserving protection)
----------------------------------------
A byT5 corrector needs the WHOLE phrase as context -- fragmenting a line into
isolated tokens starves it (it then "fixes" commas to periods, etc.). So we:

  1. send the whole line/phrase through the corrector with full context;
  2. token-align the model's OUTPUT back to the INPUT (order-preserving);
  3. for every input token that is PROTECTED (contains a digit, or is a
     name-initial like "K.", "A.", "J:r"), RESTORE the original token verbatim,
     overriding whatever the model did to it.

This keeps the LM's strength (fixing å/ä/ö and letter confusions, with context)
while guaranteeing incomes, tax sums, house numbers, years and initials are
never altered. Ordinary words -- including capitalized surnames/place names --
flow through the model (so "Goteborg" -> "Göteborg" still works); only the
strictly numeric / initial spans are frozen, because those are the data that
must not be "improved".

Public API
----------
``Corrector(model_id=KBLAB_ID, viklofg_fallback=True)`` -- lazy-loads the model.
``corrector.correct_text(text) -> text`` -- corrects a multi-line block
    line-by-line and, within a line, phrase-by-phrase with span protection.
``correct_text(text) -> text`` -- module-level convenience over a shared default
    Corrector.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cache-only at run time

import re
from typing import List, Optional

KBLAB_ID = "KBLab/swedish-ocr-correction"
VIKLOFG_ID = "viklofg/swedish-ocr-correction"

# A token is "protected" -- restored verbatim after correction (never allowed to
# be altered by the LM) if it is DATA, i.e. it:
#   - contains any digit  (incomes, tax sums, house numbers, years), OR
#   - is a name-initial    ("K.", "A.", "J:r").
# NOTE: ordinary/capitalized words (surnames, place names) are NOT frozen -- they
# flow through the model so its å/ä/ö + letter-confusion fixes apply; only the
# numeric/initial DATA spans are protected.
_HAS_DIGIT = re.compile(r"\d")
_INITIAL = re.compile(r"^[A-ZÅÄÖ][.:][A-Za-zÅÄÖåäö]?$|^[A-ZÅÄÖ][.:]?$")
# split a line into word tokens (for alignment) keeping non-space groups
_WORD_RE = re.compile(r"\S+")


def is_protected_token(tok: str) -> bool:
    """True if ``tok`` is DATA that must be restored verbatim after correction
    (a number or a name-initial). Capitalized words are NOT protected here."""
    core = tok.strip(",.;:()[]\"'«»")
    if not core:
        return False  # pure punctuation: let the model normalize spacing
    if _HAS_DIGIT.search(core):
        return True
    if _INITIAL.match(tok) or _INITIAL.match(core):
        return True
    return False


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _ned_tok(a: str, b: str) -> float:
    """Cheap normalized edit distance between two short tokens (for restore
    matching). Avoids a hard dependency on the editdistance package."""
    a, b = a or "", b or ""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


class Corrector:
    """Lazy CPU wrapper around the KBLab Swedish OCR corrector with span
    protection. Falls back to viklofg if KBLab cannot be loaded."""

    def __init__(
        self,
        model_id: str = KBLAB_ID,
        viklofg_fallback: bool = True,
        max_bytes: int = 110,
        num_beams: int = 4,
        num_threads: Optional[int] = None,
        min_corr_chars: int = 6,
        max_drift: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.viklofg_fallback = viklofg_fallback
        self.max_bytes = max_bytes
        self.num_beams = num_beams
        self.num_threads = num_threads
        self.min_corr_chars = min_corr_chars
        self.max_drift = max_drift
        self._model = None
        self._tok = None
        self._loaded_id: Optional[str] = None

    # -- loading ---------------------------------------------------------- #
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        else:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

        last_err: Optional[Exception] = None
        ids = [self.model_id]
        if self.viklofg_fallback and self.model_id != VIKLOFG_ID:
            ids.append(VIKLOFG_ID)
        for mid in ids:
            try:
                model = T5ForConditionalGeneration.from_pretrained(mid)
                # KBLab ships its own ByT5Tokenizer; viklofg uses google/byt5-small.
                try:
                    tok = AutoTokenizer.from_pretrained(mid)
                except Exception:
                    tok = AutoTokenizer.from_pretrained("google/byt5-small")
                model.eval()
                self._model, self._tok, self._loaded_id = model, tok, mid
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"could not load any corrector ({ids}): {last_err}")

    @property
    def loaded_id(self) -> Optional[str]:
        return self._loaded_id

    # -- raw model call --------------------------------------------------- #
    def _generate(self, phrase: str) -> str:
        import torch

        self._load()
        enc = self._tok(
            phrase, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_length=512, num_beams=self.num_beams
            )
        return self._tok.decode(out[0], skip_special_tokens=True).strip()

    def _accept(self, src: str, out: str) -> str:
        """Hallucination guard. A byT5 corrector run on very short or
        garbage-shaped input can FABRICATE fluent newspaper text. Reject the
        model output and keep the source when it drifted too far:
          * too short to correct meaningfully (< min_corr_chars alpha chars), OR
          * the model rewrote it into something dissimilar (NED to source above
            ``max_drift``) AND substantially longer (a fabrication, not a fix).
        Otherwise accept the correction."""
        if not out:
            return src
        d = _ned_tok(src.strip(), out.strip())
        longer = len(out) > len(src) * 1.5 + 8
        if d > self.max_drift and (longer or len(src.strip()) < 6):
            return src
        return out

    # -- correct one whole phrase (byte-chunked) with FULL context -------- #
    def _correct_phrase(self, phrase: str) -> str:
        phrase = phrase.strip()
        if not phrase:
            return phrase
        # Skip inputs too short / too sparse to correct (a lone "A.", "AA,",
        # "—"): the model has no signal and tends to hallucinate. Need at least
        # ``min_corr_chars`` alphabetic characters.
        n_alpha = sum(ch.isalpha() for ch in phrase)
        if n_alpha < self.min_corr_chars:
            return phrase
        # Chunk by byte budget so byt5's ~128-byte window is never exceeded,
        # but keep each chunk as long as possible (context preserves accuracy).
        if _byte_len(phrase) <= self.max_bytes:
            try:
                fixed = self._generate(phrase)
            except Exception:
                return phrase
            return self._accept(phrase, fixed or phrase)
        words = phrase.split(" ")
        out: List[str] = []
        cur: List[str] = []
        cur_bytes = 0
        for w in words:
            wb = _byte_len(w) + 1
            if cur and cur_bytes + wb > self.max_bytes:
                out.append(self._correct_phrase(" ".join(cur)))
                cur, cur_bytes = [], 0
            cur.append(w)
            cur_bytes += wb
        if cur:
            out.append(self._correct_phrase(" ".join(cur)))
        return " ".join(out)

    # -- public: correct a LINE, then RESTORE protected data spans -------- #
    def correct_line(self, line: str) -> str:
        """Correct the whole line with full context, then token-align the output
        back to the input and restore every PROTECTED (numeric / initial) token
        verbatim -- so the LM fixes the prose but never edits the data."""
        if not line or not line.strip():
            return line
        src_tokens = _WORD_RE.findall(line)
        # collect which source tokens are protected data spans
        protected = [t for t in src_tokens if is_protected_token(t)]

        corrected = self._correct_phrase(line.strip())
        if not protected:
            return re.sub(r"[ \t]{2,}", " ", corrected).strip()

        out_tokens = _WORD_RE.findall(corrected)
        # Order-preserving restore: walk source tokens; for each protected one,
        # find the best-matching output token at/after the current pointer and
        # overwrite it with the verbatim original. This survives the LM
        # inserting/deleting nearby words.
        from difflib import SequenceMatcher

        sm = SequenceMatcher(a=src_tokens, b=out_tokens, autojunk=False)
        result = list(out_tokens)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            # a protected source token in [i1,i2) that the model changed:
            for si in range(i1, i2):
                st = src_tokens[si]
                if not is_protected_token(st):
                    continue
                # map to the parallel output slot; clamp into the changed block
                if j1 < j2:
                    # pick the output token in [j1,j2) most similar to st, else first
                    cand_idx = min(
                        range(j1, j2),
                        key=lambda jj: _ned_tok(st, out_tokens[jj]),
                    )
                    result[cand_idx] = st
                else:
                    # model deleted it -> re-insert at j1
                    result.insert(min(j1, len(result)), st)
        restored = " ".join(result)
        return re.sub(r"[ \t]{2,}", " ", restored).strip()

    def correct_text(self, text: str) -> str:
        """Correct a multi-line block line-by-line (the public entry point)."""
        if not text:
            return text
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return "\n".join(self.correct_line(ln) for ln in lines)


# --------------------------------------------------------------------------- #
# module-level convenience
# --------------------------------------------------------------------------- #
_DEFAULT: Optional[Corrector] = None


def correct_text(text: str) -> str:
    """Correct text with a shared default KBLab corrector (lazy-loaded)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Corrector()
    return _DEFAULT.correct_text(text)


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    import io, sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    c = Corrector()
    # protection unit checks (no model needed): only DATA spans are protected
    assert is_protected_token("12") and is_protected_token("12.500")
    assert is_protected_token("K.") and is_protected_token("A:r")
    assert is_protected_token("4") and is_protected_token("12,500")
    assert not is_protected_token("Andersson")   # surnames flow through the LM
    assert not is_protected_token("Goteborg")
    assert not is_protected_token("direktor")
    assert not is_protected_token("belagen")
    print("token-protection checks OK")

    # noisy Swedish strings (general + a directory-style line with name+income)
    samples = [
        "Stockholrns stad ar belagen vid Malaren",
        "Goteborg och Malmo ar stora stader i Sverige",
        "Andersson, Karl, direktor, inkomst 12 500 kronor",
        "Bergstrom, Anna, lararinna vid folkskolan, 4 200",
    ]
    print(f"\n=== KBLab smoke (loaded: pending) ===")
    for s in samples:
        fixed = c.correct_text(s)
        print(f"NOISY : {s}")
        print(f"FIXED : {fixed}")
        print("-")
    print("loaded_id:", c.loaded_id)
    # confirm name + number survived verbatim
    out = c.correct_text("Andersson, Karl, direktor, inkomst 12 500 kronor")
    assert "Andersson" in out and "12 500" in out, out
    print("name+number preserved OK")
    print("SMOKE_OK")


if __name__ == "__main__":
    _smoke()
