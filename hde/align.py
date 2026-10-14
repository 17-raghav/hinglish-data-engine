"""Word-level forced alignment: MMS acoustic model + a hand-written CTC Viterbi.

Why forced alignment and not ASR timestamps: we already know the words (the
verbatim transcript); we only need *when* each was said. A CTC model gives, for
every 20 ms frame, a probability for each character. Viterbi finds the single
best path through the "trellis" that spells our exact transcript.

CTC trellis: the target "ab" is expanded to states  _ a _ b _  (blank between
and around tokens). At each frame the path may stay in its state, move to the
next one, or skip a blank (only between two *different* tokens, since "aa"
needs a blank between the two a's to be emitted twice).

Devanagari is romanised with uroman because MMS_FA's vocabulary is
Latin characters (it was trained on romanised text for 1,100+ languages).
"""
import re
from dataclasses import dataclass

import numpy as np

NEG_INF = -1e30


def ctc_viterbi(log_probs: np.ndarray, tokens: list[int], blank: int = 0):
    """Best CTC alignment of `tokens` to `log_probs` (T x V).

    Returns (path, score): path[t] = index into the expanded state list
    (even index = blank, odd index 2k+1 = tokens[k]).
    """
    T = log_probs.shape[0]
    states = [blank]
    for tok in tokens:
        states += [tok, blank]
    S = len(states)
    if T < len(tokens):
        raise ValueError(f"audio too short: {T} frames for {len(tokens)} tokens")
    states = np.array(states)
    emit = log_probs[:, states]                   # T x S

    # skip allowed into state s from s-2 when s is a token different from s-2's token
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = states[3::2] != states[1:-2:2]

    dp = np.full((T, S), NEG_INF)
    back = np.zeros((T, S), dtype=np.int8)        # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = emit[0, 0]
    dp[0, 1] = emit[0, 1]
    for t in range(1, T):
        stay = dp[t - 1]
        step = np.concatenate([[NEG_INF], dp[t - 1, :-1]])
        skip = np.concatenate([[NEG_INF, NEG_INF], dp[t - 1, :-2]])
        skip = np.where(can_skip, skip, NEG_INF)
        cand = np.stack([stay, step, skip])       # 3 x S
        back[t] = cand.argmax(0)
        dp[t] = cand.max(0) + emit[t]

    # must end in the last token or the final blank
    end = S - 1 if dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    score = dp[-1, end]
    path = np.empty(T, dtype=np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])          # int(): keep s a Python int (back is int8)
    assert path[0] in (0, 1), "a CTC path must start in the first blank or first token"
    return path, float(score)


@dataclass
class WordSpan:
    word: str
    start: float
    end: float
    score: float     # mean log-prob of the frames assigned to the word's characters


def romanize_words(words: list[str], uroman, dictionary: dict) -> list[str]:
    """Each word -> lowercase Latin letters the MMS vocabulary knows."""
    out = []
    for w in words:
        r = uroman.romanize_string(w, lcode="hin").lower()
        out.append("".join(c for c in r if c in dictionary and c != "-"))
    return out


def align_words(log_probs: np.ndarray, words: list[str], roman: list[str],
                dictionary: dict, seconds_per_frame: float) -> list[WordSpan]:
    """Align words (already romanised) and return a time span per original word.

    Words that romanise to nothing (rare symbols) get a zero-length span at the
    previous word's end, so every word still has ordered, in-bounds times.
    """
    tokens, owner = [], []          # owner[k] = which word token k belongs to
    for wi, r in enumerate(roman):
        for c in r:
            tokens.append(dictionary[c])
            owner.append(wi)
    path, _ = ctc_viterbi(log_probs, tokens, blank=0)

    first, last, scores = {}, {}, {}
    for t, s in enumerate(path):
        if s % 2 == 1:                              # a token state, not blank
            wi = owner[(s - 1) // 2]
            first.setdefault(wi, t)
            last[wi] = t
            scores.setdefault(wi, []).append(log_probs[t, tokens[(s - 1) // 2]])

    spans, prev_end = [], 0.0
    for wi, w in enumerate(words):
        if wi in first:
            st = first[wi] * seconds_per_frame
            en = (last[wi] + 1) * seconds_per_frame
            sc = float(np.mean(scores[wi]))
        else:
            st = en = prev_end
            sc = NEG_INF
        st = max(st, prev_end)
        spans.append(WordSpan(w, round(st, 3), round(max(en, st), 3), sc))
        prev_end = spans[-1].end
    return spans


class MMSAligner:
    """Wraps torchaudio's MMS_FA bundle (weights download on first use)."""

    def __init__(self, device: str = "cuda"):
        import torch
        import torchaudio
        import uroman as ur
        self.torch = torch
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=False).to(device).eval()
        self.dictionary = self.bundle.get_dict(star=None)
        self.device = device
        self.uroman = ur.Uroman()
        self.sr = self.bundle.sample_rate            # 16 kHz

    def emissions(self, wav16k: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(wav16k).float().unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            em, _ = self.model(x)
            em = self.torch.log_softmax(em, dim=-1)  # no-op if already log-probs
        return em[0].cpu().numpy()

    def align(self, wav16k: np.ndarray, text: str) -> list[WordSpan]:
        words = [w for w in re.sub(r"[।?!,.]", " ", text).split() if w]
        roman = romanize_words(words, self.uroman, self.dictionary)
        lp = self.emissions(wav16k)
        spf = len(wav16k) / self.sr / lp.shape[0]
        return align_words(lp, words, roman, self.dictionary, spf)


def alignment_flags(spans: list[WordSpan], duration: float,
                    min_score: float = -6.0) -> list[str]:
    """QC rules used to send a record to manual review."""
    flags = []
    zero = [s.word for s in spans if s.end - s.start <= 0]
    if zero:
        flags.append(f"{len(zero)} zero-length words")
    low = [s.word for s in spans if s.score < min_score]
    if low:
        flags.append(f"{len(low)} low-confidence words")
    if spans and spans[-1].end > duration + 1e-3:
        flags.append("word ends after clip")
    # > 1.5 s for a single word usually means a pause was swallowed into it
    long = [s.word for s in spans if s.end - s.start > 1.5]
    if long:
        flags.append(f"{len(long)} words longer than 1.5 s")
    return flags
