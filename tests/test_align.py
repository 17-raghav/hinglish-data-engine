import itertools

import numpy as np

from hde.align import align_words, ctc_viterbi


def brute_force(lp, tokens, blank=0):
    states = [blank]
    for t in tokens:
        states += [t, blank]
    S, T = len(states), lp.shape[0]
    best = (-np.inf, None)
    for path in itertools.product(range(S), repeat=T):
        if path[0] not in (0, 1) or path[-1] not in (S - 1, S - 2):
            continue
        ok = True
        for a, b in zip(path, path[1:]):
            d = b - a
            if d not in (0, 1, 2) or (d == 2 and (b % 2 == 0 or states[b] == states[b - 2])):
                ok = False
                break
        if ok:
            sc = sum(lp[t, states[s]] for t, s in enumerate(path))
            if sc > best[0]:
                best = (sc, path)
    return best


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for tokens in ([1, 2], [1, 1], [2, 3, 2]):
        for _ in range(5):
            lp = np.log(rng.dirichlet(np.ones(4), size=7))
            path, score = ctc_viterbi(lp, tokens)
            bscore, _ = brute_force(lp, tokens)
            assert np.isclose(score, bscore)


def test_align_words_times_are_ordered_and_in_bounds():
    # 10 frames, clear peaks: 'a' at frames 1-2, 'b' at frames 6-7
    V, T = 3, 10
    lp = np.full((T, V), np.log(0.05))
    lp[:, 0] = np.log(0.9)
    for t in (1, 2):
        lp[t] = np.log([0.05, 0.9, 0.05])
    for t in (6, 7):
        lp[t] = np.log([0.05, 0.05, 0.9])
    spans = align_words(lp, ["अ", "ब", "।"], ["a", "b", ""], {"a": 1, "b": 2}, 0.02)
    assert [s.word for s in spans] == ["अ", "ब", "।"]
    assert spans[0].start == 0.02 and spans[0].end == 0.06
    assert spans[1].start == 0.12 and spans[1].end == 0.16
    assert spans[2].start == spans[2].end == spans[1].end     # unalignable word


def test_long_sequence_no_overflow():
    """Real sentences have hundreds of CTC states; the backtrack index must not wrap."""
    rng = np.random.default_rng(1)
    tokens = list(rng.integers(1, 28, size=150))
    T = 600
    lp = np.log(rng.dirichlet(np.ones(29), size=T))
    path, score = ctc_viterbi(lp, tokens)
    assert path[-1] in (2 * len(tokens), 2 * len(tokens) - 1)
    assert all(0 <= b - a <= 2 for a, b in zip(path, path[1:]))
