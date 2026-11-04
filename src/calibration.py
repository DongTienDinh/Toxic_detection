"""Validation-only class-bias calibration for macro-F1 decision rules."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def macro_f1_score(labels: Sequence[int], predicted: Sequence[int]) -> float:
    truth = np.asarray(labels, dtype=np.int64)
    guess = np.asarray(predicted, dtype=np.int64)
    if truth.shape != guess.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("labels and predicted must be nonempty aligned vectors")
    scores = []
    for label in range(3):
        true_positive = int(((truth == label) & (guess == label)).sum())
        false_positive = int(((truth != label) & (guess == label)).sum())
        false_negative = int(((truth == label) & (guess != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def apply_class_biases(logits: np.ndarray, biases: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits)
    offsets = np.asarray(biases, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("logits must have shape [N,3]")
    if offsets.shape != (3,) or not np.isfinite(offsets).all():
        raise ValueError("biases must contain three finite values")
    return values + offsets[None, :]


def fit_macro_f1_biases(
    logits: np.ndarray,
    labels: Sequence[int],
    *,
    radius: float = 1.2,
    step: float = 0.05,
) -> dict:
    """Fit OFFENSIVE/HATE logit offsets on validation data only.

    CLEAN is fixed to zero because adding the same constant to all three logits
    cannot change argmax. Ties prefer the smaller bias norm, then lexical order.
    """
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape != (truth.size, 3) or truth.size == 0:
        raise ValueError("Expected aligned nonempty logits [N,3] and labels [N]")
    if not np.isfinite(values).all() or not np.isin(truth, [0, 1, 2]).all():
        raise ValueError("logits/labels contain invalid values")
    if radius <= 0 or step <= 0:
        raise ValueError("radius and step must be positive")

    candidates = np.arange(-radius, radius + step / 2, step, dtype=np.float64)
    best_biases = np.zeros(3, dtype=np.float64)
    raw_score = macro_f1_score(truth, values.argmax(axis=-1))
    best_key = (raw_score, 0.0, 0.0, 0.0)
    for offensive_bias in candidates:
        for hate_bias in candidates:
            biases = np.array([0.0, offensive_bias, hate_bias], dtype=np.float64)
            score = macro_f1_score(truth, (values + biases).argmax(axis=-1))
            key = (
                score,
                -float(np.dot(biases, biases)),
                -abs(float(offensive_bias)),
                -abs(float(hate_bias)),
            )
            if key > best_key:
                best_key = key
                best_biases = biases
    return {
        "biases": [float(value) for value in best_biases],
        "raw_macro_f1": raw_score,
        "calibrated_macro_f1": float(best_key[0]),
        "radius": float(radius),
        "step": float(step),
        "source": "validation_only",
    }
