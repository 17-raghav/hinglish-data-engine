import pytest

from hde.budget import BudgetExceeded, Ledger, stt_cost, tts_cost
from hde.sarvam_api import Sarvam


class FakeTTS:
    def __init__(self):
        self.n = 0

    def convert(self, **kw):
        self.n += 1
        import base64
        return type("R", (), {"audios": [base64.b64encode(b"RIFF").decode()]})()


class FakeClient:
    def __init__(self):
        self.text_to_speech = FakeTTS()


def test_prices():
    assert stt_cost(60) == pytest.approx(0.49)
    assert tts_cost("x" * 1000) == pytest.approx(3.0)


def test_cap_blocks_calls_and_cache_is_free(tmp_path):
    api = Sarvam(tmp_path, cap=1.0)
    api._client = FakeClient()
    api.tts("क" * 200, "hi-IN", "ritu")              # ₹0.60
    api.tts("क" * 200, "hi-IN", "ritu")              # cached: free, no call
    assert api._client.text_to_speech.n == 1
    assert api.ledger.spent == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded):
        api.tts("ख" * 200, "hi-IN", "ritu")          # would reach ₹1.20 > ₹1.00
    assert api._client.text_to_speech.n == 1
    assert Ledger(tmp_path / "spend.json", 1.0).spent == pytest.approx(0.6)   # persisted
