"""Hard spending cap for Sarvam API credits (1 credit = ₹1).

Prices from https://www.sarvam.ai/api-pricing (checked 2026-10-14, pay-as-you-go):
  speech to text   ₹0.49 per minute of audio
  text to speech   ₹3.00 per 1,000 characters
  translate        ₹2.00 per 1,000 characters
  transliterate    ₹2.00 per 1,000 characters
  Sarvam 105B      ₹15 per 1M input tokens, ₹60 per 1M output tokens

Before every call we estimate its cost and refuse it if total spend would pass
the cap. After the call the actual cost (token usage for chat) is recorded.
The ledger lives in a JSON file, so the cap holds across runs and restarts;
cached (archived) calls cost nothing and are never counted.
"""
import json
from pathlib import Path

STT_PER_MIN = 0.49
TTS_PER_CHAR = 3.00 / 1000
TEXT_PER_CHAR = 2.00 / 1000
CHAT_IN_PER_TOK = 15 / 1e6
CHAT_OUT_PER_TOK = 60 / 1e6


class BudgetExceeded(RuntimeError):
    pass


def stt_cost(seconds: float) -> float:
    return seconds / 60 * STT_PER_MIN


def tts_cost(text: str) -> float:
    return len(text) * TTS_PER_CHAR


def text_cost(text: str) -> float:
    return len(text) * TEXT_PER_CHAR


def chat_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * CHAT_IN_PER_TOK + completion_tokens * CHAT_OUT_PER_TOK


def chat_estimate(messages, max_out=1500) -> float:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return chat_cost(chars // 2, max_out)          # pessimistic: Devanagari ~2 chars/token


class Ledger:
    def __init__(self, path, cap: float):
        self.path, self.cap = Path(path), cap
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"spent": 0.0, "by_kind": {}}

    @property
    def spent(self) -> float:
        return self.data["spent"]

    def check(self, kind: str, estimate: float):
        if self.spent + estimate > self.cap:
            raise BudgetExceeded(f"{kind}: would spend ₹{estimate:.3f} on top of ₹{self.spent:.2f}; cap is ₹{self.cap:.2f}")

    def record(self, kind: str, cost: float):
        self.data["spent"] = round(self.spent + cost, 4)
        self.data["by_kind"][kind] = round(self.data["by_kind"].get(kind, 0.0) + cost, 4)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")

    def summary(self) -> str:
        parts = ", ".join(f"{k} ₹{v:.2f}" for k, v in sorted(self.data["by_kind"].items()))
        return f"Sarvam spend ₹{self.spent:.2f} of ₹{self.cap:.2f} cap ({parts})"
