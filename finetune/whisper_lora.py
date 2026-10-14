"""LoRA fine-tune Whisper-small on the Part A verbatim transcripts and report
before/after WER, CER and entity accuracy on held-out data.

  python -m finetune.whisper_lora --data data --out runs/whisper_lora

Choices worth defending in an interview:
  - Split by *speaker* (real) so the test set has unseen voices: no leakage.
  - Whisper takes at most 30 s, our clips go to 60 s. We cut at sentence
    boundaries using our own Hindi word alignment (hi_start/hi_end).
  - Both hypothesis and reference go through the SAME verbatim normaliser
    before scoring, so "2.5 लाख" vs "ढाई लाख" style differences are not
    counted as errors by accident and the metric is fair to the base model.
  - LoRA on attention projections only: ~1% of the weights train, base stays
    frozen (no forgetting, tiny checkpoint) - the same situation as Part B.
"""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

from hde.audio_io import resample
from hde.manifest import read_jsonl
from hde.normalize import normalize_verbatim
from hde.numbers_hi import parse_numbers
from hde.budget import BudgetExceeded
from hde.sarvam_api import Sarvam

SR = 16000
MAX_SEG = 28.0


def segments(rec, root):
    """Cut a record into <=28 s pieces at sentence boundaries."""
    x, sr = sf.read(str(root / rec["audio"]), dtype="float32")
    x = resample(x, sr, SR)
    out, cur = [], []
    for s in rec["sentences"] + [None]:
        if s is not None and (not cur or s["hi_end"] - cur[0]["hi_start"] <= MAX_SEG):
            cur.append(s)
            continue
        if cur:
            a = max(0.0, cur[0]["hi_start"] - 0.2)
            b = min(len(x) / SR, cur[-1]["hi_end"] + 0.2)
            if b - a <= 30.0:
                out.append({"audio": x[int(a * SR):int(b * SR)],
                            "text": " ".join(c["hi_text"] for c in cur),
                            "rec": rec["id"], "synthetic": rec["source"]["synthetic"]})
        cur = [s] if s is not None else []
    return out


def split(records, seed):
    """Real: hold out 20% of speakers. Synthetic: hold out 15% of records."""
    rng = random.Random(seed)
    real_spk = sorted({r["speaker_id"] for r in records if not r["source"]["synthetic"]})
    rng.shuffle(real_spk)
    test_spk = set(real_spk[: max(1, len(real_spk) // 5)])
    train, test = [], []
    for r in records:
        if r["source"]["synthetic"]:
            (test if rng.random() < 0.15 else train).append(r)
        else:
            (test if r["speaker_id"] in test_spk else train).append(r)
    return train, test


def entity_match(ref, hyp):
    key = lambda t: Counter(round(v, 2) for v in parse_numbers(t) if v != 1.0)
    return key(ref) == key(hyp)


def evaluate(model, processor, segs, norm, device, batch=8):
    model.eval()
    hyps = []
    for i in range(0, len(segs), batch):
        chunk = segs[i:i + batch]
        feats = processor.feature_extractor([s["audio"] for s in chunk], sampling_rate=SR,
                                            return_tensors="pt").input_features.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                    enabled=device == "cuda"):
            ids = model.generate(input_features=feats, language="hi", task="transcribe",
                                 max_new_tokens=225)
        hyps += processor.batch_decode(ids, skip_special_tokens=True)
    strip = lambda t: re.sub(r"[।?!,]", "", t).strip()
    refs = [strip(norm(s["text"])) for s in segs]
    hyps_n = [strip(norm(h)) if h.strip() else "" for h in hyps]
    ent = [(r, h) for r, h, s in zip(refs, hyps_n, segs) if [v for v in parse_numbers(s["text"]) if v != 1.0]]
    by = lambda flag: [(r, h) for r, h, s in zip(refs, hyps_n, segs) if s["synthetic"] == flag]
    res = {"segments": len(segs),
           "wer": jiwer.wer(refs, hyps_n), "cer": jiwer.cer(refs, hyps_n),
           "entity_accuracy": sum(entity_match(r, h) for r, h in ent) / max(1, len(ent)),
           "entity_segments": len(ent)}
    for name, flag in (("real", False), ("synthetic", True)):
        pairs = by(flag)
        if pairs:
            res[f"wer_{name}"] = jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])
    examples = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps_n))[:15]]
    return res, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--archive", default="archive")
    ap.add_argument("--out", default="runs/whisper_lora")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--max-credits", type=float, default=15.0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = Sarvam(args.archive, cap=args.max_credits)   # transliterates Latin words in hypotheses (cached)

    def translit(latin):
        try:
            return api.transliterate_to_devanagari(latin)
        except BudgetExceeded:
            return latin            # scored as-is: counts as an error, never silently fixed
    norm = lambda t: normalize_verbatim(t, translit)

    records = read_jsonl(root / "manifest.jsonl")
    train_r, test_r = split(records, args.seed)
    train = [s for r in train_r for s in segments(r, root)]
    test = [s for r in test_r for s in segments(r, root)]
    print(f"train {len(train)} segments ({len(train_r)} records), test {len(test)} ({len(test_r)})")

    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language="hindi", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    sot = model.config.decoder_start_token_id

    before, ex_before = evaluate(model, processor, test, norm, device)
    print("before:", before)

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch):
        feats = processor.feature_extractor([b["audio"] for b in batch], sampling_rate=SR,
                                            return_tensors="pt").input_features
        ids = [processor.tokenizer(b["text"]).input_ids[:448] for b in batch]
        ids = [i[1:] if i and i[0] == sot else i for i in ids]   # model prepends SOT itself
        L = max(map(len, ids))
        labels = torch.full((len(ids), L), -100, dtype=torch.long)
        for k, i in enumerate(ids):
            labels[k, :len(i)] = torch.tensor(i)
        return feats, labels

    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True,
                                         collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.01)
    total = args.epochs * len(loader)
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    log = open(out / "train_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "epoch", "loss", "lr", "seconds"])
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for feats, labels in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_features=feats.to(device), labels=labels.to(device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            w.writerow([step, epoch, round(loss.item(), 4), sched.get_last_lr()[0], round(time.time() - t0, 1)])
            if step % 20 == 0:
                print(f"step {step}/{total} loss {loss.item():.3f}")
                log.flush()
    log.close()

    after, ex_after = evaluate(model.get_base_model(), processor, test, norm, device)
    print("after:", after)
    model.save_pretrained(out / "adapter")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    allp = sum(p.numel() for p in model.parameters())
    result = {"model": args.model, "lora_rank": args.rank, "epochs": args.epochs, "lr": args.lr,
              "trainable_params": trainable, "total_params": allp,
              "train_segments": len(train), "test_segments": len(test),
              "train_minutes": round(time.time() - t0, 1) / 60,
              "before": before, "after": after}
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "examples.json").write_text(json.dumps({"before": ex_before, "after": ex_after},
                                                  indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
