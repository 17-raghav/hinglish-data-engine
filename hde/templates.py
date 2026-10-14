"""Parallel Hindi/English templates for entity-dense synthetic speech.

Why templates instead of an LLM: the entities are sampled in code, so a template
gives a Hindi line AND its English target with every amount, account digit, date
and name correct by construction - no translation step, no LLM cost, nothing to
hallucinate. (The first version asked Sarvam-105B to write scripts; the reasoning
model spent its whole token budget thinking and returned empty text.)

Hindi keeps everyday English words in Devanagari (बैंक, अकाउंट, पेमेंट), as the PS
requires for code-mixed speech.
"""
import random

from .numbers_hi import digits_to_words, to_words

NAMES = {"रमेश शर्मा": "Ramesh Sharma", "प्रिया वर्मा": "Priya Verma", "अंकित गुप्ता": "Ankit Gupta",
         "सुनीता यादव": "Sunita Yadav", "मोहम्मद आसिफ़": "Mohammed Asif", "हरप्रीत सिंह": "Harpreet Singh",
         "कविता नायर": "Kavita Nair", "राहुल मेहता": "Rahul Mehta", "पूजा अग्रवाल": "Pooja Agarwal",
         "विकास चौहान": "Vikas Chauhan"}
PLACES = {"लखनऊ": "Lucknow", "पटना": "Patna", "जयपुर": "Jaipur", "इंदौर": "Indore", "पंचकूला": "Panchkula",
          "कानपुर": "Kanpur", "भोपाल": "Bhopal", "नागपुर": "Nagpur", "देहरादून": "Dehradun", "रांची": "Ranchi"}
MONTHS = list(zip("जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(),
                  "January February March April May June July August September October November December".split()))

# (domain, Hindi with {amt}, English with {amt})
AMOUNT_LINES = [
    ("bank", "मेरे बैंक अकाउंट से {amt} रुपये कट गए हैं।", "{amt} rupees have been deducted from my bank account."),
    ("loan", "मुझे {amt} रुपये का लोन चाहिए।", "I need a loan of {amt} rupees."),
    ("insurance", "मेरा इंश्योरेंस क्लेम {amt} रुपये का है।", "My insurance claim is for {amt} rupees."),
    ("delivery", "मैंने {amt} रुपये का ऑर्डर किया था।", "I had placed an order of {amt} rupees."),
    ("booking", "टिकट के लिए मैंने {amt} रुपये पेमेंट किए।", "I paid {amt} rupees for the ticket."),
    ("doctor", "डॉक्टर की फ़ीस {amt} रुपये बताई गई।", "The doctor's fee was said to be {amt} rupees."),
    ("recharge", "मैंने {amt} रुपये का रिचार्ज करवाया।", "I got a recharge of {amt} rupees done."),
    ("rent", "इस महीने का किराया {amt} रुपये है।", "This month's rent is {amt} rupees."),
    ("salary", "मेरी सैलरी के {amt} रुपये आने थे।", "My salary of {amt} rupees was supposed to come."),
    ("bill", "बिजली का बिल {amt} रुपये आया है।", "The electricity bill has come to {amt} rupees."),
]

EN_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen nineteen").split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def to_words_en(n: int) -> str:
    """Integer -> English words in the Indian system (lakh, crore)."""
    if n < 20:
        return EN_UNITS[n]
    if n < 100:
        return EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + EN_UNITS[n % 10])
    parts = []
    for scale, name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand"), (100, "hundred")):
        q, n = divmod(n, scale)
        if q:
            parts.append(f"{to_words_en(q)} {name}")
    if n:
        parts.append(to_words_en(n))
    return " ".join(parts)


def digits_en(s: str) -> str:
    return " ".join(EN_UNITS[int(c)] for c in s)


def build(spec: dict, rng: random.Random):
    """Returns (hindi_sentences, english_sentences, dropped_values_per_sentence)."""
    name_hi, place_hi = spec["name"], spec["place"]
    name_en, place_en = NAMES[name_hi], PLACES[place_hi]
    day, (month_hi, month_en) = spec["day"], MONTHS[spec["month"]]
    amt_hi, amt_en = spec["amount_words"], to_words_en(int(spec["amount"]))
    dropped = []
    if spec["self_correction"]:
        amt_hi = f"{spec['wrong_words']}, मतलब {amt_hi}"      # speaker corrects themself
        dropped = [spec["wrong_amount"]]
    _, line_hi, line_en = next(l for l in AMOUNT_LINES if l[0] == spec["domain"])

    hi = [f"हाँ तो मैं {name_hi} बोल रहा हूँ, {place_hi} से।",
          line_hi.format(amt=amt_hi),
          f"ये {to_words(day)} {month_hi} को हुआ था।"]
    en = [f"Yes, this is {name_en} speaking, from {place_en}.",
          line_en.format(amt=amt_en),
          f"This happened on {month_en} {to_words_en(day)}."]
    drops = [[], dropped, []]
    if "digits" in spec:
        hi.append(f"मेरा अकाउंट नंबर {digits_to_words(spec['digits'])} है।")
        en.append(f"My account number is {digits_en(spec['digits'])}.")
        drops.append([])
    if spec["negation"]:
        hi.append("पर अभी तक पैसे वापस नहीं आए।")
        en.append("But the money has not come back yet.")
        drops.append([])
    if spec["condition"]:
        hi.append("अगर आज काम नहीं हुआ तो मैं कंप्लेंट करूँगा।")
        en.append("If the work is not done today, then I will complain.")
        drops.append([])
    hi.append("प्लीज़ जल्दी देखिए।")
    en.append("Please look into it quickly.")
    drops.append([])
    return hi, en, drops
