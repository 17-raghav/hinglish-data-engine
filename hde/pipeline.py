"""Part A data engine CLI.

  python -m hde.pipeline real     --n 300   # IndicVoices Hindi clips -> data/stage_real.jsonl
  python -m hde.pipeline synth    --n 150   # entity-dense synthetic clips -> data/stage_synth.jsonl
  python -m hde.pipeline finalize           # verbatim text, alignment, translation, validation
  python -m hde.pipeline validate           # re-check data/manifest.jsonl

Add --offline to any stage to rebuild purely from archive/ (no API calls).
"""
import argparse
import io
import json
import random
from pathlib import Path

import soundfile as sf

from . import manifest as M
from .audio_io import SR, resample, to_mono_float, write_wav
from .normalize import normalize_verbatim, split_sentences
from .budget import BudgetExceeded
from .sarvam_api import Sarvam, TTS_MODEL
from .synth import make_script, sample_spec, synthesize
from .entities import check_pair
from .translate import translate_sentences

TEXT_KEYS = ["verbatim", "text", "transcript", "transcription", "sentence", "normalized"]
SPK_KEYS = ["speaker_id", "speaker", "spk_id", "user_id"]


def _pick(row, keys, default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def stage_real(args, api):
    """Stream IndicVoices Hindi and keep clips of 5-60 s (PS sessions are 10-60 s)."""
    from datasets import Audio, load_dataset
    ds = load_dataset("ai4bharat/IndicVoices", "hindi", split=args.split, streaming=True)
    # the audio column's name varies by dataset (IndicVoices: "audio_filepath"); find it by type
    acol = next(k for k, f in ds.features.items() if isinstance(f, Audio))
    ds = ds.cast_column(acol, Audio(decode=False))      # raw bytes; we decode ourselves (deterministic)
    out, root = [], Path(args.data)
    skipped = 0
    (root / "audio").mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(ds):
        if i == 0:
            print("columns:", list(row.keys()))
        if len(out) >= args.n:
            break
        audio = row.get(acol)
        if not audio or not (audio.get("bytes") or audio.get("path")):
            skipped += 1                              # some rows ship without audio
            if skipped % 50 == 0:
                print(f"skipped {skipped} rows with no audio so far")
            continue
        try:
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32") if audio.get("bytes") \
                else sf.read(audio["path"], dtype="float32")
        except Exception as e:                        # noqa: BLE001 - corrupt/unsupported file
            skipped += 1
            print(f"row {i}: unreadable audio ({type(e).__name__}), skipped")
            continue
        x = resample(to_mono_float(x), sr, SR)
        dur = len(x) / SR
        text = _pick(row, TEXT_KEYS)
        if not text or not 5.0 <= dur <= 60.0:
            continue
        rid = f"rec-r{len(out):05d}"
        write_wav(root / "audio" / f"{rid}.wav", x)
        src_file = audio.get("path") or f"{args.split}-row{i}"     # the clip's file inside the dataset
        out.append({"id": rid, "audio": f"audio/{rid}.wav", "duration": round(dur, 3),
                    "speaker_id": f"iv-{_pick(row, SPK_KEYS, 'unk')}",
                    "source": {"corpus": f"ai4bharat/IndicVoices:hindi:{args.split}",
                               "source_file": src_file, "start": 0.0, "end": round(dur, 3),
                               "synthetic": False},
                    "raw_text": text})
    M.write_jsonl(root / "stage_real.jsonl", out)
    print(f"kept {len(out)} real clips, skipped {skipped} rows without usable audio")


def stage_synth(args, api):
    rng = random.Random(args.seed)
    root = Path(args.data)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    out, rejected = [], []
    for i in range(args.n):
      try:
        spec = sample_spec(rng)
        hi_sents, en_sents, drops = make_script(spec, rng)
        text = " ".join(hi_sents)
        x, info = synthesize(api, text, i, rng)
        if x is None:
            rejected.append({"i": i, "reason": f"tts_{info['reason']}", "info": info, "text": text})
            continue
        dur = len(x) / SR
        if dur > 60:
            rejected.append({"i": i, "reason": "too_long"})
            continue
        rid = f"rec-s{i:05d}"
        write_wav(root / "audio" / f"{rid}.wav", x)
        out.append({"id": rid, "audio": f"audio/{rid}.wav", "duration": round(dur, 3),
                    "speaker_id": f"bulbul-{info['voice']}",
                    "source": {"corpus": "synthetic:parallel-templates", "source_file": f"audio/{rid}.wav",
                               "start": 0.0, "end": round(dur, 3), "synthetic": True,
                               "model": TTS_MODEL, "voice": info["voice"]},
                    "raw_text": text,
                    "synth": {"spec": spec, "en": en_sents, "dropped": drops, **info}})
        if (i + 1) % 5 == 0:
            print(f"{i + 1}/{args.n}: kept {len(out)}. {api.ledger.summary()}")
      except BudgetExceeded as e:
        print(f"STOPPED at clip {i}: {e}")
        break
    M.write_jsonl(root / "stage_synth.jsonl", out)
    M.write_jsonl(root / "rejected_synth.jsonl", rejected)
    print(f"kept {len(out)}, rejected {len(rejected)}")


def stage_finalize(args, api):
    from .align import MMSAligner, alignment_flags
    root = Path(args.data)
    stubs = []
    for name in ("stage_real.jsonl", "stage_synth.jsonl"):
        if (root / name).exists():
            stubs += M.read_jsonl(root / name)
    aligner = MMSAligner(device=args.device)
    nllb = None                                     # loaded only if real clips need translating
    records, review, reasons, qc_notes = [], [], [], []
    for n, stub in enumerate(stubs):
      try:
        text = normalize_verbatim(stub["raw_text"], api.transliterate_to_devanagari)
        sents = split_sentences(text)
        x, sr = sf.read(str(root / stub["audio"]), dtype="float32")
        spans = aligner.align(resample(x, sr, aligner.sr), text)
        flags = alignment_flags(spans, stub["duration"])
        synth = stub.get("synth")
        if synth:                                    # template English: exact by construction
            if len(synth["en"]) != len(sents):
                translations, method = None, "parallel template (sentence count mismatch)"
            else:
                names = [synth["spec"]["name"], synth["spec"]["place"]]
                translations = [{"en_text": e, "en_alt": [], "problems": check_pair(h, e, d, names=names)}
                                for h, e, d in zip(sents, synth["en"], synth["dropped"])]
                method = "parallel Hindi/English template; number+negation+name check"
        elif args.translator == "nllb":
            if nllb is None:
                from .mt_nllb import NLLB
                nllb = NLLB(args.archive, device=args.device)
            en = nllb.translate(sents)
            translations = [{"en_text": e, "en_alt": [], "problems": check_pair(h, e)}
                            for h, e in zip(sents, en)]
            method = "NLLB-200 distilled 600M (beam 4); number+negation check"
        else:
            translations, tries = translate_sentences(api, sents)
            method = f"sarvam-105b, streaming-friendly prompt; number+negation+name check; {tries} attempt(s)"
        problems = [p for t in (translations or []) for p in t["problems"]]
        if translations is None:
            problems = ["translation failed"]
        rec = {k: stub[k] for k in ("id", "audio", "duration", "speaker_id", "source")}
        if translations:
            rec["sentences"] = M.build_sentences(sents, spans, translations)
        rec["hi_words"] = [{"w": s.word, "start": s.start, "end": s.end} for s in spans]
        rec["translation_method"] = method
        severe = [f for f in flags if f.startswith("word ends after clip")]
        low = sum(1 for sp in spans if sp.score < -6.0)
        if spans and low / len(spans) > 0.2:
            severe.append(f"{low}/{len(spans)} words low-confidence")
        if problems or severe:
            review.append({**rec, "problems": problems, "align_flags": flags})
            reasons += ["translation_check" if problems else "alignment_flag"]
            continue
        if flags:                                   # minor: keep the record, log for manual review
            qc_notes.append({"id": rec["id"], "align_flags": flags})
        bad = M.validate_record(rec, root)
        if bad:
            review.append({**rec, "problems": bad})
            reasons.append("validator")
            continue
        records.append(rec)
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(stubs)} done, kept {len(records)}. {api.ledger.summary()}")
      except BudgetExceeded as e:
        print(f"STOPPED at record {n}: {e}")
        break
    M.write_jsonl(root / "manifest.jsonl", records)
    M.write_jsonl(root / "review.jsonl", review)
    M.write_jsonl(root / "qc_notes.jsonl", qc_notes)
    rej_synth = M.read_jsonl(root / "rejected_synth.jsonl") if (root / "rejected_synth.jsonl").exists() else []
    st = M.stats(records, reasons + [r["reason"] for r in rej_synth])
    (root / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(st, indent=2, ensure_ascii=False))


def stage_validate(args, api):
    root = Path(args.data)
    recs = M.read_jsonl(root / "manifest.jsonl")
    bad = {r["id"]: M.validate_record(r, root) for r in recs}
    bad = {k: v for k, v in bad.items() if v}
    hours = M.union_hours(recs)
    print(f"{len(recs)} records, {hours:.3f} h counted, {len(bad)} invalid")
    for k, v in list(bad.items())[:20]:
        print(k, v)
    assert hours <= M.MAX_HOURS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["real", "synth", "finalize", "validate"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offline", action="store_true", help="rebuild from archive only")
    ap.add_argument("--translator", choices=["nllb", "sarvam-llm"], default="nllb",
                    help="nllb: free, local GPU (default); sarvam-llm: Sarvam-105B, paid")
    ap.add_argument("--max-credits", type=float, default=15.0,
                    help="hard cap on total Sarvam spend in ₹ (credits), across all runs")
    args = ap.parse_args()
    api = Sarvam(args.archive, offline=args.offline, cap=args.max_credits)
    {"real": stage_real, "synth": stage_synth, "finalize": stage_finalize,
     "validate": stage_validate}[args.stage](args, api)
    print(f"Sarvam API calls this run: {api.calls}. {api.ledger.summary()}")
    if args.stage == "real":
        # HF streaming leaves download threads alive that block interpreter exit;
        # every output is already written and flushed, so exit hard.
        import os, sys
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
