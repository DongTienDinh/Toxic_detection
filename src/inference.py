"""Offline inference from a complete best-model artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import ID2LABEL, stable_softmax
from .models import load_model_artifact


class ToxicityClassifier:
    def __init__(self, artifact_dir: str | Path, *, device: str | None = None):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.artifact_dir = Path(artifact_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir, local_files_only=True
        )
        self.model = load_model_artifact(self.artifact_dir)
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model.to(self.device).eval()

    def predict(self, texts: Sequence[str], *, batch_size: int = 16, max_length: int = 128) -> list[dict]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain nonempty strings")
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(
                    list(texts[start:start + batch_size]),
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                chunks.append(self.model(**encoded).logits.detach().float().cpu().numpy())
        logits = np.concatenate(chunks, axis=0)
        probabilities = stable_softmax(logits)
        predictions = logits.argmax(axis=1)
        return [
            {
                "label_id": int(predictions[index]),
                "label": ID2LABEL[int(predictions[index])],
                "probabilities": {
                    ID2LABEL[label]: float(probabilities[index, label]) for label in (0, 1, 2)
                },
            }
            for index in range(len(texts))
        ]
