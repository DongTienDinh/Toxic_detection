"""Fixed-label metrics for ViHSD training and final evaluation."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = (0, 1, 2)
ID2LABEL = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def _arrays(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if truth.size == 0:
        raise ValueError("Metrics require at least one sample")
    if not set(np.unique(truth)).issubset(LABELS):
        raise ValueError("y_true contains labels outside 0/1/2")
    if not set(np.unique(prediction)).issubset(LABELS):
        raise ValueError("y_pred contains labels outside 0/1/2")
    return truth, prediction


def detailed_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    truth, prediction = _arrays(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(truth, prediction, labels=LABELS)
    row_total = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_total,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_total != 0,
    )
    clean = truth == 0
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "clean_fpr": float((prediction[clean] != 0).mean()) if clean.any() else None,
        "hate_recall": float(recall[2]),
        "per_class": {
            ID2LABEL[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_row_normalized": normalized.tolist(),
    }


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    prediction = np.argmax(np.asarray(logits), axis=-1)
    result = detailed_metrics(labels, prediction)
    return {
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "accuracy": result["accuracy"],
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected logits shaped [N,3]")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")
    return probabilities


def prediction_rows(
    *,
    sidecar: Sequence[Mapping],
    logits: np.ndarray,
    experiment_id: str,
    seed: int,
    checkpoint_hash: str,
    protocol_version: str,
) -> list[dict]:
    values = np.asarray(logits)
    if len(sidecar) != len(values):
        raise ValueError("Sidecar and logits length mismatch")
    probabilities = stable_softmax(values)
    predicted = values.argmax(axis=1)
    rows = []
    seen = set()
    for index, source in enumerate(sidecar):
        sample_id = source["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id {sample_id}")
        seen.add(sample_id)
        rows.append({
            "sample_id": sample_id,
            "y_true": int(source["labels"]),
            "y_pred": int(predicted[index]),
            "logit_clean": float(values[index, 0]),
            "logit_offensive": float(values[index, 1]),
            "logit_hate": float(values[index, 2]),
            "p_clean": float(probabilities[index, 0]),
            "p_offensive": float(probabilities[index, 1]),
            "p_hate": float(probabilities[index, 2]),
            "experiment_id": experiment_id,
            "seed": int(seed),
            "checkpoint_hash": checkpoint_hash,
            "protocol_version": protocol_version,
        })
    return rows
