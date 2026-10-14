# Hinglish Speech Data Engine + Whisper LoRA

A miniature of Inter IIT 15.0 / Sarvam "High Prep" Part A: it turns Hindi speech
(code-mixed with English) into records in the exact competition format (PS §5.2),
then LoRA-fine-tunes Whisper-small on the verbatim transcripts.

Flow:

1. **Sources**: IndicVoices Hindi clips + synthetic clips (entities sampled in code → parallel Hindi/English templates → Bulbul TTS → Saaras round-trip QC)
2. **Verbatim normalisation**: numbers to spoken words (lakh/crore), English words to Devanagari, sentence-final punctuation
3. **Hindi word alignment**: MMS acoustic model + own CTC Viterbi
4. **Translation**: real clips by NLLB-200 (free, local GPU, deterministic); synthetic clips carry their exact template English
5. **Checks**: number + negation + name check on every sentence, alignment QC flags, §5.2 validator
6. **Output**: `manifest.jsonl` + `stats.json`, then Whisper-small LoRA trained on it (before/after scores)

## Run

Everything that needs a GPU or the network runs in `notebooks/run_colab.ipynb` (T4 is enough).
Offline tests run anywhere: `python -m pytest -q tests` (54 tests, no API, no GPU).

| Command | What it does |
|---|---|
| `python -m hde.pipeline real --n 300` | Streams IndicVoices Hindi (CC-BY-4.0, gated), keeps 5–60 s clips, resamples to 24 kHz PCM16 |
| `python -m hde.pipeline synth --n 150` | Entity-dense synthetic clips with round-trip QC |
| `python -m hde.pipeline finalize` | Normalise, align, translate, check, validate → `data/manifest.jsonl`, `review.jsonl`, `stats.json` |
| `python -m hde.pipeline validate` | Re-checks every §5.2 rule and the 200 h cap |
| `... --offline` | Rebuild from `archive/` only, zero API calls |
| `python -m finetune.whisper_lora` | LoRA fine-tune + before/after WER, CER, entity accuracy |

## Design decisions

- **Numbers are the product.** `hde/numbers_hi.py` converts both ways in the Indian system, including
  डेढ़ / ढाई / साढ़े / सवा / पौने and digit-by-digit account numbers. Tested by round-tripping 0 to 999,999,999.
- **Entities are checked, not trusted.** Every translation passes `entities.check_pair`: numbers on both sides
  must match except values the translator declares abandoned in a self-correction; a negated Hindi
  sentence must produce a negated English one; every person/place name must survive. Failures are retried once
  with the complaint, then sent to review.
- **Names across scripts** (`hde/names.py`) reuse the approach from my Business Entity Matching project: reduce
  both sides to a script-free consonant skeleton ("हरप्रीत सिंह" → uroman "harapriit simh" → `hrprt sn`, same as
  "Harpreet Singh"), then rapidfuzz with an 85 threshold. True pairs score 100, wrong names 43–73.
- **Why not an LLM for translation.** The first version used Sarvam-105B. It is a reasoning model: under a
  1,500-token cap it spent the whole budget thinking and returned empty text for all 49 requests (₹4.6 spent,
  nothing usable). Now empty answers are never cached, and the default translator is NLLB-200 distilled 600M on
  the Colab GPU: free, about a second per clip, deterministic with beam search. `--translator sarvam-llm` keeps
  the LLM path (4,096-token budget). Synthetic clips need no translation at all: their English comes from the
  same template as the Hindi, so every amount and name is exact.
- **Synthetic audio is verified.** TTS skips, repeats and misreads numbers, so each clip is re-transcribed by
  Saaras (verbatim mode); reject on CER > 12% or any number mismatch; retry with another voice.
- **Forced alignment is written by hand.** `hde/align.py` implements the CTC Viterbi over the blank-expanded
  trellis (tested against brute force); MMS_FA supplies frame log-probs over uroman-romanised Devanagari.
- **Credit-capped.** `hde/budget.py` prices every Sarvam call before making it (published per-minute / per-character /
  per-token rates) and refuses any call that would pass `--max-credits`; the ledger persists across runs and cached
  calls are free.
- **Reproducible.** Every API response is archived under a hash of its request; resampling is deterministic.
  `--offline` rebuilds the dataset with no API calls.
- **Fair evaluation.** Test split holds out *speakers*; hypotheses and references go through the same normaliser.

## Results (run on 2026-10-14, Colab T4)

| Dataset | Value |
|---|---|
| Valid records | **101** (89 IndicVoices Hindi from 27 speakers + 12 synthetic), 0 validator errors |
| Hindi words with timings | 2,581 (16 min of audio, mean clip 9.6 s) |
| Real translations rejected by entity checks | **30 / 120 (25%)** — NLLB changed a number, negation or name |
| Synthetic clips passing Saaras round-trip QC | 12 / 12 (first voice) |
| Total Sarvam spend | ₹11.84 (₹4.63 of it the failed LLM attempt) |

| Whisper-small, held-out speakers (22 clips) | Before | After LoRA (run 1 / run 2) |
|---|---|---|
| WER | 0.956 | **0.492 / 0.497** |
| CER | 0.597 | **0.217 / 0.220** |
| WER, real IndicVoices only | 0.972 | 0.503 / 0.510 |

Two independent training runs agree within 0.005 WER (the gap is GPU fp16 nondeterminism), so the
improvement is not a lucky seed.

LoRA r=32 on q/k/v/out projections: 7.1M of 249M parameters (2.8%), 10 epochs = 100 steps, 2.2 min.
Caveats: small test set; part of the gain is learning verbatim conventions (numbers as words,
Devanagari spelling), which is exactly what the Part A transcripts require. Entity accuracy is not
reported because only 3 test clips contain numbers.

Colab note: if `peft` fails with an incompatible `torchao` version, run `pip uninstall -y torchao` (unused here).
