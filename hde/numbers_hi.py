"""Hindi numbers in the Indian system (lakh, crore), both directions.

to_words(250000)              -> "दो लाख पचास हज़ार"
digits_to_words("9876")       -> "नौ आठ सात छह"        (account / phone numbers)
parse_numbers("ढाई लाख रुपये") -> [250000.0]           (handles डेढ़, ढाई, साढ़े, सवा, पौने)

Hindi 0-99 are irregular words, so they are a lookup table, not rules.
"""
import re
import unicodedata

UNITS = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(UNITS) == 100

# Common alternative spellings seen in real transcripts.
VARIANTS = {"पांच": 5, "छः": 6, "छै": 6, "छे": 6, "तिरपन": 53, "तिरसठ": 63,
            "सरसठ": 67, "उनासी": 79, "नवासी": 89, "पचानवे": 95, "इकावन": 51,
            "चव्वालीस": 44, "उन्नासी": 79, "सत्तानबे": 97}
WORD2NUM = {w: i for i, w in enumerate(UNITS)} | VARIANTS

SCALES = {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 10**5,
          "करोड़": 10**7, "करोड": 10**7, "अरब": 10**9}
FRACTIONS = {"डेढ़": 1.5, "डेढ": 1.5, "ढाई": 2.5}            # standalone values
MODIFIERS = {"साढ़े": 0.5, "साढे": 0.5, "सवा": 0.25, "पौने": -0.25}  # apply to next number
POINT = {"दशमलव", "प्वाइंट", "पॉइंट"}


def to_words(n: int) -> str:
    """Integer -> Hindi words in the Indian system."""
    if n < 0:
        return "माइनस " + to_words(-n)
    if n < 100:
        return UNITS[n]
    parts = []
    for scale, name in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        q, n = divmod(n, scale)
        if q:
            parts.append(f"{to_words(q)} {name}")  # crore can exceed 99: recurse
    if n:
        parts.append(UNITS[n])
    return " ".join(parts)


def digits_to_words(s: str) -> str:
    """'9876' -> 'नौ आठ सात छह' (how account and phone numbers are read)."""
    return " ".join(UNITS[int(c)] for c in s if c.isdigit())


def decimal_to_words(s: str) -> str:
    whole, frac = s.split(".")
    return f"{to_words(int(whole or 0))} दशमलव {digits_to_words(frac)}"


def nfc(text: str) -> str:
    """One spelling per character: precomposed ज़ (U+095B) and ज + nukta become the same."""
    return unicodedata.normalize("NFC", text)


def _clean(tok: str) -> str:
    return re.sub(r"[।,.?!\"'()]", "", tok)


def parse_numbers(text: str) -> list[float]:
    """Find every number spoken in Hindi words and return its value.

    Runs of 3+ single digits ('नौ आठ सात') are read as a digit string
    (account/phone number) and returned as that integer.
    """
    values, group = [], []
    for tok in nfc(text).split() + ["<END>"]:
        t = _clean(tok)
        if t in WORD2NUM or t in SCALES or t in FRACTIONS or t in MODIFIERS or t in POINT:
            group.append(t)
        elif group:
            values.extend(_value_of(group))
            group = []
    return values


def _value_of(group: list[str]) -> list[float]:
    # Digit-by-digit reading: every token is a single digit.
    if len(group) >= 3 and all(WORD2NUM.get(t, 99) < 10 for t in group):
        return [float("".join(str(WORD2NUM[t]) for t in group))]
    if any(t in POINT for t in group):
        i = next(i for i, t in enumerate(group) if t in POINT)
        whole = _value_of(group[:i]) or [0.0]
        frac = "".join(str(WORD2NUM[t]) for t in group[i + 1:] if WORD2NUM.get(t, 99) < 10)
        return [whole[0] + (float("0." + frac) if frac else 0.0)]
    total, current, mod, seen = 0.0, 0.0, 0.0, False
    for t in group:
        if t in MODIFIERS:
            mod = MODIFIERS[t]
        elif t in FRACTIONS:
            current += FRACTIONS[t]
            seen = True
        elif t in WORD2NUM:
            current += WORD2NUM[t] + mod
            mod, seen = 0.0, True
        elif t == "सौ":
            current = (current or (1 + mod)) * 100
            mod, seen = 0.0, True
        elif t in SCALES:
            total += (current or (1 + mod)) * SCALES[t]
            current, mod, seen = 0.0, 0.0, True
    return [total + current] if seen else []


# ---------- written text -> spoken words (for verbatim transcripts) ----------
_NUM_RE = re.compile(r"(₹\s?)?(\d[\d,]*(?:\.\d+)?)(\s?%)?")


def verbalize_numbers(text: str) -> str:
    """Replace digits in a Hindi sentence with spoken words.

    Heuristic (documented, not perfect):
      - Indian-comma or currency amounts and numbers up to 6 digits -> cardinal words
      - 7+ plain digits (account, phone) -> read digit by digit
      - '₹' becomes 'रुपये' after the number, '%' becomes 'प्रतिशत'
    """
    def repl(m):
        rupee, num, pct = m.group(1), m.group(2), m.group(3)
        plain = num.replace(",", "")
        if "." in plain:
            words = decimal_to_words(plain)
        elif len(plain) >= 7 and "," not in num and not rupee:
            words = digits_to_words(plain)
        else:
            words = to_words(int(plain))
        if rupee:
            words += " रुपये"
        if pct:
            words += " प्रतिशत"
        return words
    return _NUM_RE.sub(repl, text)
