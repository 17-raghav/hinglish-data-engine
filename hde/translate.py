"""Hindi -> English sentence translations designed as *streaming* targets.

Design choices (and the interview answer for each):
  - Keep English close to Hindi order where English grammar allows: a target
    that reorders heavily forces the speech model to wait longer.
  - Keep the speaker's units (lakh, crore), numbers as digits or words.
  - Resolve self-corrections to the final value; the translator must *declare*
    abandoned values in `dropped` so the entity checker can account for them.
  - Return a literal variant too, saved as en_alt.

Every output is checked by entities.check_pair; a failing translation is retried
once with the checker's complaint, then sent to review (never silently kept).
"""
import json
import re

from .entities import check_pair

SYSTEM = """You translate spoken Hindi (often mixed with English) into English for a live speech interpreter.
Return ONLY JSON: {"sentences": [{"en": "...", "en_literal": "...", "dropped": [numbers], "names": ["..."]}]} with exactly one item per input sentence, in order.

Rules:
- "en": natural spoken English. Keep it close to the Hindi word order whenever English allows, short and plain.
- "en_literal": a more literal rendering.
- Write every number in words ("two point five lakh rupees", "account nine eight seven six"). Keep lakh and crore as units; never convert to millions.
- Keep every number, name, date, account digit and negation exactly.
- Self-corrections ("तीन लाख, नहीं, साढ़े तीन लाख"): translate only the corrected meaning. Put the abandoned numeric values in "dropped" (e.g. [300000]); otherwise "dropped": [].
- Fillers (अं, हम्म) are dropped; repetitions are said once.
- English words spoken in the Hindi stay as the same English words.
- "names": every person, place or organisation name in the Hindi sentence, copied exactly as written in the Hindi (Devanagari); [] if none."""


def _parse(raw: str, n: int):
    raw = re.sub(r"```(?:json)?", "", raw)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        items = json.loads(raw[a:b + 1])["sentences"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(items, list) or len(items) != n:
        return None
    out = []
    for it in items:
        if not isinstance(it, dict) or not str(it.get("en", "")).strip():
            return None
        dropped = [float(x) for x in it.get("dropped") or [] if isinstance(x, (int, float))]
        names = [n for n in it.get("names") or [] if isinstance(n, str)]
        out.append({"en": it["en"].strip(), "en_literal": str(it.get("en_literal", "")).strip(),
                    "dropped": dropped, "names": names})
    return out


def translate_sentences(api, hi_sentences: list[str], known_names: list[str] | None = None):
    """known_names: names we put into synthetic scripts ourselves (ground truth).
    Returns (list of dicts with en_text, en_alt, problems), number of attempts."""
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(hi_sentences))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": numbered}]
    result, feedback = None, None
    for attempt in range(2):
        msgs = messages if feedback is None else messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Fix these problems and return the full JSON again: {feedback}"}]
        try:
            raw = api.chat(msgs, temperature=0.2, seed_tag=f"try{attempt}")
        except Exception as e:                      # EmptyResponse: model gave no text
            if type(e).__name__ != "EmptyResponse":
                raise
            raw, feedback = "", "empty response"
            continue
        parsed = _parse(raw, len(hi_sentences))
        if parsed is None:
            feedback = "invalid JSON or wrong number of sentences"
            continue
        result = []
        for hi, p in zip(hi_sentences, parsed):
            probs = check_pair(hi, p["en"], p["dropped"], names=p["names"] + (known_names or []))
            alts = [p["en_literal"]] if p["en_literal"] and p["en_literal"] != p["en"] else []
            result.append({"en_text": p["en"], "en_alt": alts, "problems": probs,
                           "dropped": p["dropped"]})
        bad = {i + 1: r["problems"] for i, r in enumerate(result) if r["problems"]}
        if not bad:
            return result, attempt + 1
        feedback = json.dumps(bad, ensure_ascii=False)
    return result, 2
