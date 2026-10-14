"""Entity-dense synthetic Hindi speech.

Real conversational audio rarely contains "ढाई लाख" or a ten-digit account number
in the same clip, yet those are exactly what entity accuracy (18 pts) tests. So:

 1. Code samples the entities (we know the ground-truth values).
 2. Parallel Hindi/English templates (hde/templates.py) turn them into a code-mixed
    Hindi call and its exact English target, sometimes with a negation, a
    condition or a self-correction.
 3. Bulbul speaks it in Hindi with a random voice and pace.
 4. Round-trip QC: Saaras re-transcribes the audio; reject if the character error
    rate is high or any number came out different (TTS skips, repeats and
    misreads numbers). Retry with another voice, then give up.
"""
import random
import re

import jiwer

from .audio_io import read_wav_bytes, resample, trim_silence, SR
from .normalize import normalize_verbatim
from .numbers_hi import digits_to_words, parse_numbers, to_words

HI_VOICES = ["aditya", "rahul", "rohan", "amit", "dev", "varun", "kabir", "ritu",
             "priya", "neha", "pooja", "simran", "kavya", "ishita", "shreya", "roopa"]
FRACTION_FORMS = [(1.5, "डेढ़"), (2.5, "ढाई")]


def sample_amount(rng: random.Random):
    """Returns (value, spoken Hindi words) covering the hard Indian forms."""
    kind = rng.choice(["plain", "lakh", "frac", "sadhe", "crore", "thousand"])
    if kind == "frac":
        v, w = rng.choice(FRACTION_FORMS)
        scale, sw = rng.choice([(1000, "हज़ार"), (10**5, "लाख"), (10**7, "करोड़")])
        return v * scale, f"{w} {sw}"
    if kind == "sadhe":
        n = rng.randint(3, 9)
        scale, sw = rng.choice([(100, "सौ"), (1000, "हज़ार"), (10**5, "लाख")])
        return (n + 0.5) * scale, f"साढ़े {to_words(n)} {sw}"
    v = {"plain": rng.randint(11, 999),
         "thousand": rng.randint(2, 99) * 1000 + rng.choice([0, 0, 500]),
         "lakh": rng.randint(1, 99) * 10**5 + rng.choice([0, 25000, 50000, 75000]),
         "crore": rng.randint(1, 20) * 10**7}[kind]
    return float(v), to_words(v)


def sample_spec(rng: random.Random) -> dict:
    from .templates import AMOUNT_LINES, MONTHS, NAMES, PLACES
    amount, amount_words = sample_amount(rng)
    spec = {"domain": rng.choice([d for d, _, _ in AMOUNT_LINES]), "amount": amount,
            "amount_words": amount_words, "name": rng.choice(sorted(NAMES)),
            "place": rng.choice(sorted(PLACES)), "day": rng.randint(1, 28),
            "month": rng.randrange(len(MONTHS)),
            "negation": rng.random() < 0.4, "condition": rng.random() < 0.3,
            "self_correction": rng.random() < 0.3}
    if rng.random() < 0.5:
        digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([4, 6, 10])))
        spec["digits"], spec["digits_words"] = digits, digits_to_words(digits)
    if spec["self_correction"]:
        wrong, wrong_words = sample_amount(rng)
        while wrong == amount:
            wrong, wrong_words = sample_amount(rng)
        spec["wrong_amount"], spec["wrong_words"] = wrong, wrong_words
    return spec


def expected_values(spec) -> list[float]:
    vals = [spec["amount"]]
    if "digits" in spec:
        vals.append(float(spec["digits"]))
    if spec["self_correction"]:
        vals.append(spec["wrong_amount"])
    return vals


def _same_numbers(a: list[float], b: list[float]) -> bool:
    key = lambda v: sorted(round(x, 2) for x in v if x != 1.0)
    return key(a) == key(b)


def make_script(spec, rng):
    """Template -> (normalised Hindi sentences, English sentences, dropped values per sentence)."""
    from .templates import build
    hi, en, drops = build(spec, rng)
    hi = [normalize_verbatim(h, lambda latin: latin) for h in hi]   # templates are all-Devanagari
    return hi, en, drops


def synthesize(api, text, idx, rng: random.Random, max_cer=0.12):
    """TTS + round-trip check. Returns (audio_24k, info) or (None, info with reason)."""
    tried = []
    for attempt in range(3):
        voice = rng.choice([v for v in HI_VOICES if v not in tried])
        tried.append(voice)
        pace = round(rng.uniform(0.9, 1.15), 2)
        wav = api.tts(text, "hi-IN", voice, pace=pace, temperature=0.6, tag=f"{idx}")
        x, sr = read_wav_bytes(wav)
        x = trim_silence(resample(x, sr, SR), SR)
        back = api.stt_verbatim(wav, audio_id=f"synth-{idx}-{voice}-{pace}")
        back = normalize_verbatim(back, api.transliterate_to_devanagari)
        cer = jiwer.cer(re.sub(r"[।?!,]", "", text), re.sub(r"[।?!,]", "", back))
        info = {"voice": voice, "pace": pace, "cer": round(cer, 4), "attempt": attempt + 1,
                "asr_back": back}
        if cer <= max_cer and _same_numbers(parse_numbers(text), parse_numbers(back)):
            return x, info
        info["reason"] = "cer" if cer > max_cer else "numbers"
    return None, info
