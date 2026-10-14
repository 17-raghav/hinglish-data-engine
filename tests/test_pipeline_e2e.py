"""synth -> finalize -> validate with a fake Sarvam API and a fake aligner (no network, no GPU)."""
import io
import json
import re
import sys
import types
from argparse import Namespace

import numpy as np
import soundfile as sf

from hde import manifest as M
from hde import pipeline as P
from hde.align import WordSpan


class FakeAPI:
    """Echoes scripts back through 'TTS' and 'STT' so round-trip QC passes."""
    def __init__(self):
        self.calls = 0
        self.ledger = types.SimpleNamespace(summary=lambda: "fake")
        self.last_tts = ""

    def chat(self, messages, temperature=0.2, seed_tag=""):
        user = messages[1]["content"]
        if "Must contain exactly these phrases" in user:
            phrases = re.findall(r'"([^"]+)"', user.split("Must contain exactly these phrases:")[1].split("\n")[0])
            return json.dumps({"text": "हाँ तो " + " और ".join(phrases) + " नहीं चाहिए।"}, ensure_ascii=False)
        from hde.names import romanize
        from hde.numbers_hi import parse_numbers
        hi = [re.sub(r"^\d+\. ", "", l) for l in user.splitlines() if re.match(r"\d+\. ", l)]
        sents = []
        for h in hi:      # a "perfect" translator: every name romanised, every number as digits, negation kept
            nums = " ".join(str(v) for v in parse_numbers(h))
            sents.append({"en": f"{romanize(h)} not {nums}", "en_literal": "", "dropped": [], "names": []})
        return json.dumps({"sentences": sents})

    def transliterate_to_devanagari(self, text):
        return "शब्द"

    def tts(self, text, lang, voice, pace=1.0, temperature=0.6, tag=""):
        self.last_tts = text
        x = 0.1 * np.sin(np.arange(int(24000 * 3)) / 10)
        buf = io.BytesIO(); sf.write(buf, x, 24000, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def stt_verbatim(self, wav, audio_id=""):
        return self.last_tts


class FakeAligner:
    sr = 16000

    def __init__(self, device="cpu"):
        pass

    def align(self, wav16k, text):
        words = [w for w in re.sub(r"[।?!,.]", " ", text).split() if w]
        dur = len(wav16k) / 16000
        step = dur / (len(words) + 1)
        return [WordSpan(w, round(i * step, 3), round((i + 0.8) * step, 3), -1.0) for i, w in enumerate(words)]


class FakeNLLB:
    """Stand-in for the GPU model: a 'perfect' translator for the test sentence."""
    def __init__(self, *a, **k):
        pass

    def translate(self, sents):
        return ["I did not get two lakh fifty thousand rupees." for _ in sents]


def test_synth_finalize_validate(tmp_path, monkeypatch):
    import hde.align, hde.mt_nllb
    monkeypatch.setattr(hde.align, "MMSAligner", FakeAligner)
    monkeypatch.setattr(hde.mt_nllb, "NLLB", FakeNLLB)
    api = FakeAPI()
    args = Namespace(data=str(tmp_path), archive=str(tmp_path / "archive"), n=6, seed=15,
                     device="cpu", split="valid", translator="nllb")
    P.stage_synth(args, api)
    stubs = M.read_jsonl(tmp_path / "stage_synth.jsonl")
    assert len(stubs) == 6, M.read_jsonl(tmp_path / "rejected_synth.jsonl")
    # one "real" clip translated by the (fake) NLLB path
    from hde.audio_io import write_wav
    write_wav(tmp_path / "audio" / "rec-r00000.wav", 0.1 * np.sin(np.arange(24000 * 4) / 7))
    M.write_jsonl(tmp_path / "stage_real.jsonl", [{
        "id": "rec-r00000", "audio": "audio/rec-r00000.wav", "duration": 4.0, "speaker_id": "iv-1",
        "source": {"corpus": "ai4bharat/IndicVoices:hindi:valid", "source_file": "x.wav",
                   "start": 0.0, "end": 4.0, "synthetic": False},
        "raw_text": "मुझे 2,50,000 रुपये नहीं मिले"}])
    P.stage_finalize(args, api)
    recs = M.read_jsonl(tmp_path / "manifest.jsonl")
    review = M.read_jsonl(tmp_path / "review.jsonl")
    assert len(recs) == 7, review[:1]
    for r in recs:
        assert M.validate_record(r, tmp_path) == []
    real = next(r for r in recs if not r["source"]["synthetic"])
    assert real["sentences"][0]["hi_text"] == "मुझे दो लाख पचास हज़ार रुपये नहीं मिले।"
    assert "NLLB" in real["translation_method"]
    syn = next(r for r in recs if r["source"]["synthetic"])
    assert "template" in syn["translation_method"] and syn["source"]["corpus"] == "synthetic:parallel-templates"
    st = json.loads((tmp_path / "stats.json").read_text())
    assert st["records"] == 7 and st["synthetic_records"] == 6


def test_templates_pass_their_own_checks():
    """Every template pair must pass the same entity checks the pipeline applies."""
    import random
    from hde.entities import check_pair
    from hde.synth import make_script, sample_spec
    rng = random.Random(0)
    for _ in range(300):
        spec = sample_spec(rng)
        hi, en, drops = make_script(spec, rng)
        for h, e, d in zip(hi, en, drops):
            assert check_pair(h, e, d, names=[spec["name"], spec["place"]]) == [], (h, e)
