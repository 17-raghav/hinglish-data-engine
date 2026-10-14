"""Pipeline pieces with a fake API: zero network, zero GPU."""
import json
import random

import numpy as np

from hde import manifest as M
from hde.align import WordSpan
from hde.audio_io import write_wav
from hde.normalize import check_verbatim, normalize_verbatim, split_sentences
from hde.synth import sample_spec, expected_values
from hde.numbers_hi import parse_numbers
from hde.translate import translate_sentences

FAKE_TRANSLIT = {"account": "अकाउंट", "please": "प्लीज़", "bank": "बैंक"}


def translit(s):
    return " ".join(FAKE_TRANSLIT.get(w.lower(), "शब्द") for w in s.split())


def test_normalize_rules():
    t = normalize_verbatim("मेरा account [noise] नंबर 9876543210 है. please जल्दी करो", translit)
    assert t == "मेरा अकाउंट नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है। प्लीज़ जल्दी करो।"
    assert check_verbatim(t) == []
    assert len(split_sentences(t)) == 2


class FakeAPI:
    def __init__(self, replies):
        self.replies = replies

    def chat(self, messages, temperature=0.2, seed_tag=""):
        return self.replies.pop(0)


def test_translate_retries_on_wrong_amount():
    hi = ["मैंने ढाई लाख रुपये भेजे।"]
    wrong = json.dumps({"sentences": [{"en": "I sent two lakh rupees.", "en_literal": "", "dropped": []}]})
    right = json.dumps({"sentences": [{"en": "I sent two point five lakh rupees.",
                                       "en_literal": "I two and a half lakh rupees sent.", "dropped": []}]})
    out, tries = translate_sentences(FakeAPI([wrong, right]), hi)
    assert tries == 2 and out[0]["problems"] == []
    assert out[0]["en_alt"] == ["I two and a half lakh rupees sent."]


def test_spec_values_are_spoken_forms():
    rng = random.Random(0)
    for _ in range(200):
        spec = sample_spec(rng)
        assert parse_numbers(spec["amount_words"]) == [spec["amount"]]
        if "digits" in spec:
            assert parse_numbers(spec["digits_words"]) == [float(spec["digits"])]
        assert len(expected_values(spec)) >= 1


def _record(tmp_path, start=0.0, end=2.0, synthetic=False):
    write_wav(tmp_path / "a.wav", np.zeros(48000, dtype=np.float32))
    sents = ["मैंने खाता बंद किया।"]
    spans = [WordSpan(w, 0.1 + 0.3 * i, 0.35 + 0.3 * i, -1.0) for i, w in enumerate(["मैंने", "खाता", "बंद", "किया"])]
    rec = {"id": "rec-1", "audio": "a.wav", "duration": 2.0, "speaker_id": "s1",
           "source": {"corpus": "c", "source_file": "f", "start": start, "end": end, "synthetic": synthetic},
           "sentences": M.build_sentences(sents, spans, [{"en_text": "I closed the account.", "en_alt": []}]),
           "hi_words": [{"w": s.word, "start": s.start, "end": s.end} for s in spans],
           "translation_method": "test"}
    return rec


def test_validator_accepts_good_and_rejects_bad(tmp_path):
    rec = _record(tmp_path)
    assert M.validate_record(rec, tmp_path) == []
    assert rec["sentences"][0]["hi_start"] == 0.1 and rec["sentences"][0]["hi_end"] == 1.25
    rec["hi_words"][2]["start"] = 5.0
    assert M.validate_record(rec, tmp_path)
    rec = _record(tmp_path, synthetic=True)
    assert "synthetic record missing source.model" in M.validate_record(rec, tmp_path)


def test_union_hours_counts_overlap_once(tmp_path):
    a = _record(tmp_path, 0, 40)
    b = _record(tmp_path, 30, 60)
    b["duration"] = 30
    assert abs(M.union_hours([a, b]) - 60 / 3600) < 1e-9
