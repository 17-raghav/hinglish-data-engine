"""Part A record format (PS section 5.2), the validator, and dataset statistics."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from .entities import is_negated_hi
from .normalize import check_verbatim
from .numbers_hi import parse_numbers

MAX_CLIP = 60.0
MAX_HOURS = 200.0


def build_sentences(hi_sentences, spans, translations):
    """Attach hi_start/hi_end from word spans to each sentence."""
    out, k = [], 0
    for hi, tr in zip(hi_sentences, translations):
        n = len(hi.replace("।", " ").replace("?", " ").replace("!", " ").replace(",", " ").split())
        ws = spans[k:k + n]
        k += n
        out.append({"hi_text": hi, "hi_start": ws[0].start if ws else 0.0,
                    "hi_end": ws[-1].end if ws else 0.0,
                    "en_text": tr["en_text"], "en_alt": tr["en_alt"]})
    return out


def validate_record(rec: dict, root: Path) -> list[str]:
    p = []
    for key in ("id", "audio", "duration", "source", "sentences", "hi_words", "translation_method"):
        if key not in rec:
            p.append(f"missing {key}")
    if p:
        return p
    src = rec["source"]
    for key in ("corpus", "source_file", "start", "end", "synthetic"):
        if key not in src or src[key] in ("", None):
            p.append(f"empty source.{key}")
    if src.get("synthetic"):
        for key in ("model", "voice"):
            if not src.get(key):
                p.append(f"synthetic record missing source.{key}")
    wav = root / rec["audio"]
    if not wav.exists():
        return p + ["audio file missing"]
    info = sf.info(str(wav))
    if info.samplerate != 24000 or info.channels != 1 or info.subtype != "PCM_16":
        p.append(f"audio must be 24 kHz mono PCM_16, got {info.samplerate} {info.channels}ch {info.subtype}")
    dur = info.frames / info.samplerate
    if abs(dur - rec["duration"]) > 0.01:
        p.append(f"duration {rec['duration']} != file {dur:.3f}")
    if dur > MAX_CLIP:
        p.append("clip longer than 60 s")
    prev = 0.0
    for w in rec["hi_words"]:
        if not (0 <= w["start"] <= w["end"] <= dur + 1e-3) or w["start"] < prev - 1e-6:
            p.append(f"bad word time {w}")
            break
        prev = w["start"]
    n_words = sum(len(s["hi_text"].replace("।", " ").replace("?", " ").replace("!", " ")
                      .replace(",", " ").split()) for s in rec["sentences"])
    if n_words != len(rec["hi_words"]):
        p.append(f"{len(rec['hi_words'])} word times for {n_words} transcript words")
    for s in rec["sentences"]:
        p += [f"hi_text: {x}" for x in check_verbatim(s["hi_text"])]
        if not s["en_text"].strip():
            p.append("empty en_text")
    return p


def union_hours(records) -> float:
    """PS counting rule: real audio = union of source intervals; synthetic = every file."""
    by_file = defaultdict(list)
    synth = 0.0
    for r in records:
        s = r["source"]
        if s["synthetic"]:
            synth += r["duration"]
        else:
            by_file[(s["corpus"], s["source_file"])].append((s["start"], s["end"]))
    real = 0.0
    for ivs in by_file.values():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s > cur_e:
                real += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        real += cur_e - cur_s
    return (real + synth) / 3600


def stats(records, rejections) -> dict:
    sents = [s for r in records for s in r["sentences"]]
    nums = sum(len(parse_numbers(s["hi_text"])) for s in sents)
    return {
        "records": len(records),
        "hours_counted": round(union_hours(records), 3),
        "real_records": sum(not r["source"]["synthetic"] for r in records),
        "synthetic_records": sum(r["source"]["synthetic"] for r in records),
        "speakers": len({r["speaker_id"] for r in records}),
        "sentences": len(sents),
        "hindi_words": sum(len(r["hi_words"]) for r in records),
        "number_mentions": nums,
        "negated_sentences": sum(is_negated_hi(s["hi_text"]) for s in sents),
        "mean_clip_s": round(sum(r["duration"] for r in records) / max(1, len(records)), 2),
        "rejections": dict(Counter(rejections)),
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
