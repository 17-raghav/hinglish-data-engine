"""Does a Hindi name survive translation into English?

Reuses the idea from my Business Entity Matching project: normalise both sides
into a script-free key, then fuzzy-match with a tuned threshold (rapidfuzz, 85).

"हरप्रीत सिंह" romanises (uroman) to "harapriit simh", but English writes
"Harpreet Singh". Exact matching fails; even edit distance on raw strings is low.
The consonant skeleton fixes most of it:

  harapriit simh  -> h r p r t  s n      (anusvara m before h/c/k -> n)
  Harpreet Singh  -> h r p r t  s n      (ng -> n, h after a consonant dropped)

Rules: lowercase; ck -> k; anusvara-style m/n before a consonant -> n; ng -> n; drop h after
a consonant (ch, sh, kh, ph, gh, th -> c, s, k, p, g, t); keep a word's first letter
even if it is a vowel; drop other vowels; collapse doubled letters.
"""
import re
from functools import lru_cache

from rapidfuzz import fuzz

THRESHOLD = 85
VOWELS = set("aeiouy")


def skeleton_word(w: str) -> str:
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return ""
    w = w.replace("ck", "k").replace("ph", "f").replace("w", "v").replace("q", "k").replace("z", "j")
    w = re.sub(r"ng", "n", w)
    w = re.sub(r"m(?=[bcdfghjklpqrstvxz])", "n", w)   # anusvara before a consonant
    w = re.sub(r"(?<=[bcdfgjklmnpqrstvxz])h", "", w)  # aspiration / digraphs
    first, rest = w[0], "".join(c for c in w[1:] if c not in VOWELS)
    return re.sub(r"(.)\1+", r"\1", first + rest)


def skeleton(text: str) -> str:
    return " ".join(s for s in (skeleton_word(w) for w in text.split()) if s)


@lru_cache(maxsize=1)
def _uroman():
    import uroman as ur
    return ur.Uroman()


def romanize(hindi: str) -> str:
    return _uroman().romanize_string(hindi, lcode="hin")


def name_score(hindi_name: str, english_text: str) -> float:
    """Best fuzzy score of the name against any same-length word window of the English."""
    key = skeleton(romanize(hindi_name))
    n = len(key.split())
    words = re.findall(r"[A-Za-z']+", english_text)
    best = 0.0
    for i in range(max(1, len(words) - n + 1)):
        window = " ".join(words[i:i + n])
        best = max(best, fuzz.ratio(key, skeleton(window)))
    return best


def name_in_english(hindi_name: str, english_text: str, threshold=THRESHOLD) -> bool:
    return name_score(hindi_name, english_text) >= threshold
