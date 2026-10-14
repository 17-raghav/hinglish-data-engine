"""Sarvam API calls with an on-disk archive.

Every request is keyed by a SHA-256 of its parameters and its response is saved
under archive/. Re-running the pipeline reads the archive instead of calling the
API, which is how the dataset rebuilds identically despite LLM and TTS
non-determinism (PS: "rebuilds every record from the source audio plus your
archived transcripts, translations and TTS outputs").

Run with offline=True to guarantee no API call is made (rebuild mode).
"""
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

from . import budget as B

TTS_MODEL = "bulbul:v3"
STT_MODEL = "saaras:v3"          # `mode` (verbatim, codemix, ...) only applies to v3
LLM_MODEL = "sarvam-105b"


class ArchiveMiss(RuntimeError):
    pass


class EmptyResponse(RuntimeError):
    """The model returned no text (e.g. reasoning used the whole token budget). Never cached."""


class Sarvam:
    def __init__(self, archive_dir="archive", offline=False, cap=15.0):
        self.dir = Path(archive_dir)
        self.offline = offline
        self.calls = 0
        self._client = None
        # spend ledger lives next to the archive so the cap holds across runs
        self.ledger = B.Ledger(self.dir / "spend.json", cap)

    @property
    def client(self):
        if self._client is None:
            from sarvamai import SarvamAI
            self._client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        return self._client

    # ---- archive helpers ----
    def _key(self, kind, params):
        blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode()
        return self.dir / kind / (hashlib.sha256(blob).hexdigest()[:24] + ".json")

    def _cached(self, kind, params, fn, estimate, actual=None):
        """estimate: worst-case ₹ for the call (checked against the cap first).
        actual(resp) -> (₹, response_to_store): the real cost once we have the response."""
        path = self._key(kind, params)
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))["response"]
            if cached not in ("", None):           # old archives may hold empty answers: ignore them
                return cached
        if self.offline:
            raise ArchiveMiss(f"{kind} not in archive: {params}")
        self.ledger.check(kind, estimate)
        for attempt in range(4):                  # simple retry on rate limits / 5xx
            try:
                resp = fn()
                break
            except Exception:                     # noqa: BLE001 - SDK raises several types
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        self.calls += 1
        cost, resp = actual(resp) if actual else (estimate, resp)
        self.ledger.record(kind, cost)
        if resp in ("", None):
            raise EmptyResponse(f"{kind}: empty response (cost ₹{cost:.3f} recorded, not cached)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": params, "response": resp}, ensure_ascii=False),
                        encoding="utf-8")
        return resp

    # ---- APIs ----
    def chat(self, messages, temperature=0.2, seed_tag=""):
        """seed_tag lets us ask for a *different* sample of the same prompt (retries)."""
        params = {"model": LLM_MODEL, "messages": messages, "temperature": temperature,
                  "tag": seed_tag}

        def call():
            r = self.client.chat.completions(messages=messages, model=LLM_MODEL,
                                             temperature=temperature, reasoning_effort="low",
                                             max_tokens=4096)
            u = r.usage
            return {"text": r.choices[0].message.content or "",
                    "in": getattr(u, "prompt_tokens", None), "out": getattr(u, "completion_tokens", None)}

        def actual(resp):
            if resp["in"] is None or resp["out"] is None:
                return B.chat_estimate(messages), resp["text"]
            return B.chat_cost(resp["in"], resp["out"]), resp["text"]
        return self._cached("chat", params, call, B.chat_estimate(messages, max_out=4096), actual)

    def stt_verbatim(self, wav_bytes: bytes, audio_id: str):
        """Saaras v3 verbatim mode: fillers and spoken numbers kept as said."""
        params = {"model": STT_MODEL, "mode": "verbatim", "audio_sha": hashlib.sha256(wav_bytes).hexdigest(),
                  "id": audio_id}

        def call():
            r = self.client.speech_to_text.transcribe(
                file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                model=STT_MODEL, mode="verbatim", language_code="hi-IN")
            return r.transcript
        import soundfile as sf
        secs = sf.info(io.BytesIO(wav_bytes)).duration
        return self._cached("stt", params, call, B.stt_cost(secs))

    def transliterate_to_devanagari(self, text: str):
        params = {"text": text}

        def call():
            r = self.client.text.transliterate(input=text, source_language_code="en-IN",
                                               target_language_code="hi-IN")
            return r.transliterated_text
        return self._cached("translit", params, call, B.text_cost(text))

    def tts(self, text, language_code, speaker, pace=1.0, temperature=0.6,
            sample_rate=24000, tag="") -> bytes:
        """Returns WAV bytes. The archive stores the base64 audio itself."""
        params = {"model": TTS_MODEL, "text": text, "language_code": language_code,
                  "speaker": speaker, "pace": pace, "temperature": temperature,
                  "sample_rate": sample_rate, "tag": tag}

        def call():
            r = self.client.text_to_speech.convert(
                text=text, language_code=language_code, model=TTS_MODEL, speaker=speaker,
                pace=pace, temperature=temperature, speech_sample_rate=sample_rate)
            return r.audios[0]
        return base64.b64decode(self._cached("tts", params, call, B.tts_cost(text)))
