"""Audio in and out. Everything in the dataset is mono 16-bit PCM at 24 kHz.

Resampling uses scipy's polyphase filter: deterministic, so a rebuild produces
byte-identical WAVs (PS reproducibility requirement).
"""
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000


def to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    """Peak-safe float -> int16 (clips rather than wraps)."""
    return (np.clip(x, -1.0, 1.0) * 32767).round().astype(np.int16)


def write_wav(path, x: np.ndarray, sr: int = SR):
    sf.write(str(path), to_pcm16(x), sr, subtype="PCM_16")


def read_wav_bytes(b: bytes):
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    return to_mono_float(x), sr


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, to_pcm16(x), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def trim_silence(x: np.ndarray, sr: int, thresh_db=-40.0, pad=0.15) -> np.ndarray:
    """Trim leading/trailing silence from TTS output (keeps `pad` seconds)."""
    win = int(0.02 * sr)
    if len(x) < win:
        return x
    frames = len(x) // win
    rms = np.sqrt((x[: frames * win].reshape(frames, win) ** 2).mean(1) + 1e-12)
    loud = np.where(20 * np.log10(rms) > thresh_db)[0]
    if not len(loud):
        return x
    a = max(0, loud[0] * win - int(pad * sr))
    b = min(len(x), (loud[-1] + 1) * win + int(pad * sr))
    return x[a:b]
