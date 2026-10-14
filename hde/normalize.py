"""Verbatim normalisation to the PS rules (IndicVoices Level 1, "as spoken"):

  - disfluencies, repetitions, fillers: kept (we never delete words)
  - English words: written in Devanagari as spoken, no Latin in brackets
  - numbers: written as spoken words
  - every sentence ends with sentence-final punctuation (। ? !)
  - noise tags: optional, we drop them
"""
import re

from .numbers_hi import nfc, verbalize_numbers

LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*")
TAG = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)")
ALLOWED = re.compile(r"[^ऀ-ॿ\s।?!,]")      # Devanagari block + punctuation
SENT_END = re.compile(r"(?<=[।?!])\s+")


def normalize_verbatim(text: str, transliterate) -> str:
    """`transliterate(latin_text) -> devanagari_text` is injected (Sarvam API in the
    pipeline, a stub in tests)."""
    t = TAG.sub(" ", nfc(text))
    t = verbalize_numbers(t)
    t = LATIN_RUN.sub(lambda m: transliterate(m.group(0)), t)
    t = t.replace(".", "।").replace("|", "।")
    t = ALLOWED.sub(" ", t)
    t = re.sub(r"\s+([।?!,])", r"\1", t)
    t = re.sub(r"([।?!])\1+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in "।?!":
        t = t.rstrip(",") + "।"
    return t


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_END.split(text) if s.strip()]


def check_verbatim(text: str) -> list[str]:
    """Format rules the validator enforces on every hi_text."""
    problems = []
    if re.search(r"[A-Za-z]", text):
        problems.append("Latin characters present")
    if re.search(r"\d", text):
        problems.append("digits present")
    if not text or text[-1] not in "।?!":
        problems.append("no sentence-final punctuation")
    return problems
