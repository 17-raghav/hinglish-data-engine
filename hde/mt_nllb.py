"""Free, local Hindi -> English translation with NLLB-200 (distilled 600M) on the GPU.

Why: the Sarvam-105B reasoning model spent its entire output budget thinking and
returned empty text for every request, while costing credits. NLLB is a plain
encoder-decoder MT model built into transformers (no remote code), runs in about a
second per clip on a T4, and with beam search it is deterministic - so the dataset
rebuilds identically. Results are also cached under archive/nllb/.

Licence note: NLLB-200 weights are CC-BY-NC-4.0 (non-commercial). Fine for this
portfolio project; the competition says any open-weight model may be used to
build data, declared in model_card.md.
"""
import hashlib
import json
from pathlib import Path

MODEL = "facebook/nllb-200-distilled-600M"


class NLLB:
    def __init__(self, archive_dir="archive", device="cuda", beams=4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch, self.device, self.beams = torch, device, beams
        self.dir = Path(archive_dir) / "nllb"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL, src_lang="hin_Deva")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn")

    def _key(self, sentences):
        blob = json.dumps({"model": MODEL, "beams": self.beams, "s": sentences}, ensure_ascii=False)
        return self.dir / (hashlib.sha256(blob.encode()).hexdigest()[:24] + ".json")

    def translate(self, sentences: list[str]) -> list[str]:
        path = self._key(sentences)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["en"]
        enc = self.tok(sentences, return_tensors="pt", padding=True, truncation=True,
                       max_length=256).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, forced_bos_token_id=self.eng,
                                      num_beams=self.beams, max_new_tokens=256)
        en = [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
        path.write_text(json.dumps({"hi": sentences, "en": en}, ensure_ascii=False), encoding="utf-8")
        return en
