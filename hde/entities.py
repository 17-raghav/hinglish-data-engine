"""Entity and intent checks between a Hindi transcript and its English translation.

The KPI that matters most (entity accuracy, 18 pts) fails silently: a fluent
translation with the wrong amount still reads well. So every translation is
checked mechanically before it enters the dataset:

  numbers   every value spoken in Hindi must appear in English, except values the
            speaker abandoned in a self-correction (declared by the translator)
  negation  a negated Hindi sentence must produce a negated English sentence
  names     person/place names must appear in English (fuzzy, script-independent)
"""
import re
from collections import Counter

from .numbers_hi import nfc, parse_numbers

EN_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
EN_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
EN_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 10**5, "lakhs": 10**5,
             "lac": 10**5, "crore": 10**7, "crores": 10**7, "million": 10**6,
             "billion": 10**9, "k": 1000}
EN_FRAC = {"half": 0.5, "quarter": 0.25}


def parse_numbers_en(text: str) -> list[float]:
    """Numbers in English text: '2.5 lakh', '2,50,000', 'two lakh fifty thousand',
    '25k', 'nine eight seven six' (digit string)."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"(\d)\s*k\b", r"\1 k", t)
    toks = re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", t)
    values, group = [], []
    for tok in toks + ["<end>"]:
        is_num = (tok[0].isdigit() or tok in EN_UNITS or tok in EN_TENS
                  or tok in EN_SCALES or (tok in ("and", "point", "half", "a") and group))
        if is_num:
            group.append(tok)
        elif group:
            values.extend(_value_en(group))
            group = []
    return values


def _value_en(group):
    while group and group[-1] in ("and", "a", "point"):
        group = group[:-1]
    if not group:
        return []
    words = [g for g in group if g not in ("and", "a")]
    if len(words) >= 3 and all(w in EN_UNITS and EN_UNITS[w] < 10 for w in words):
        return [float("".join(str(EN_UNITS[w]) for w in words))]
    total, current, out = 0.0, 0.0, []
    i = 0
    while i < len(words):
        w = words[i]
        if w[0].isdigit():
            if current or total:          # "50,000 ... 3" => separate numbers
                out.append(total + current)
                total, current = 0.0, 0.0
            current = float(w.replace(",", ""))
        elif w == "point" and i + 1 < len(words):
            frac = ""
            while i + 1 < len(words) and words[i + 1] in EN_UNITS:
                i += 1
                frac += str(EN_UNITS[words[i]])
            current += float("0." + frac) if frac else 0
        elif w in EN_UNITS:
            current += EN_UNITS[w]
        elif w in EN_TENS:
            current += EN_TENS[w]
        elif w == "half":
            current += 0.5
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in EN_SCALES:
            total += (current or 1) * EN_SCALES[w]
            current = 0.0
        i += 1
    out.append(total + current)
    return [v for v in out if v or "zero" in words or "0" in words]


HI_NEG = {"नहीं", "नही", "न", "मत", "ना", "बिना"}
EN_NEG = re.compile(r"\b(not|no|never|nothing|none|nobody|without|neither|nor|cannot)\b|n't\b",
                    re.I)


def is_negated_hi(text: str) -> bool:
    return any(re.sub(r"[।,.?!]", "", w) in HI_NEG for w in nfc(text).split())


def is_negated_en(text: str) -> bool:
    return bool(EN_NEG.search(text))


def _norm(v: float) -> float:
    return round(v, 2)


def check_pair(hi_text: str, en_text: str, dropped: list[float] | None = None,
               names: list[str] | None = None) -> list[str]:
    """Return a list of problems; empty list means the pair passes.
    `names`: Hindi spellings of person/place/org names that must survive (see names.py)."""
    from .names import name_in_english
    problems = []
    hi = Counter(_norm(v) for v in parse_numbers(hi_text))
    for d in dropped or []:
        if hi[_norm(d)] > 0:
            hi[_norm(d)] -= 1
    hi = +hi
    en = Counter(_norm(v) for v in parse_numbers_en(en_text))
    # Hindi "एक" doubles as the article "a" ("एक दिन" = "one day" / "a day"), so a
    # lone 1 is not compared. Real amounts of one ("एक लाख") still are (100000).
    hi.pop(1.0, None)
    en.pop(1.0, None)
    if missing := hi - en:
        problems.append(f"numbers missing in English: {sorted(missing.elements())}")
    if extra := en - hi:
        problems.append(f"numbers not in Hindi: {sorted(extra.elements())}")
    if is_negated_hi(hi_text) and not is_negated_en(en_text):
        problems.append("Hindi is negated but English is not")
    for name in names or []:
        if name and nfc(name) in nfc(hi_text) and not name_in_english(name, en_text):
            problems.append(f"name missing or misspelled in English: {name}")
    return problems
