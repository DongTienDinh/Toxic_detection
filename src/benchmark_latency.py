"""Comparable model-only and end-to-end latency measurement."""

from __future__ import annotations

from statistics import median
import time
from typing import Sequence

import numpy as np

from .models import count_parameters


def _summary(milliseconds: Sequence[float], items_per_iteration: int) -> dict:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "iterations": int(len(values)),
        "p50_ms": float(median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_items_per_second": float(
            items_per_iteration / (values.mean() / 1000.0)
        ),
    }


def benchmark(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int,
    iterations: int = 200,
    warmup: int = 30,
    max_length: int = 128,
) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Protocol latency benchmark requires CUDA")
    if len(texts) < batch_size:
        raise ValueError("Not enough benchmark texts for one batch")
    device = torch.device("cuda")
    model.to(device).eval()
    batch_texts = list(texts[:batch_size])
    encoded_cpu = tokenizer(
        batch_texts, truncation=True, max_length=max_length, padding=True,
        return_tensors="pt", return_token_type_ids=False,
    )
    encoded_device = {key: value.to(device) for key, value in encoded_cpu.items()}

    with torch.inference_mode():
        for _ in range(warmup):
            model(**encoded_device)
        torch.cuda.synchronize()
        model_times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(**encoded_device)
            torch.cuda.synchronize()
            model_times.append((time.perf_counter() - start) * 1000)

        end_to_end_times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            encoded = tokenizer(
                batch_texts, truncation=True, max_length=max_length, padding=True,
                return_tensors="pt", return_token_type_ids=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            torch.softmax(logits, dim=-1)
            torch.cuda.synchronize()
            end_to_end_times.append((time.perf_counter() - start) * 1000)
    return {
        "device": torch.cuda.get_device_name(0),
        "precision": str(next(model.parameters()).dtype),
        "batch_size": batch_size,
        "max_length": max_length,
        "parameters": count_parameters(model),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "model_only": _summary(model_times, batch_size),
        "end_to_end": _summary(end_to_end_times, batch_size),
    }
