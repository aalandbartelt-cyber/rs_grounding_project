"""Translate original-training query text with the downloaded offline Qwen model."""
import json

import torch

from common import ROOT, load_training_cache, write_json
from vlm import Qwen


def main():
    torch.set_num_threads(4)
    cache = load_training_cache()
    path = ROOT / "features/translations.json"
    translations = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    model = Qwen()
    for number, text in enumerate(cache["texts"], 1):
        if text not in translations:
            prompt = ("Translate the following Chinese aircraft description into concise English. "
                      "Preserve colors, engine counts, wing and engine positions, location and motion. "
                      "Return only the English translation, without adding any visual attributes.\n" + text)
            translated = model.answer(prompt, max_tokens=128)
            if not translated or len(translated) > 1000:
                raise ValueError("Invalid model-generated translation")
            translations[text] = translated
            write_json(path, translations)
        print(f"Translated original-training query {number}/{len(cache['texts'])}", flush=True)


if __name__ == "__main__":
    main()
