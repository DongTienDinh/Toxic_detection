"""Seed aggregation and paired bootstrap comparisons."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .io_utils import write_json
from .metrics import detailed_metrics


def mean_sample_sd(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot aggregate zero values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
    }


def aggregate_metric_rows(rows: Iterable[Mapping]) -> list[dict]:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[str(row["experiment"])].append(row)
    output = []
    for experiment, group in sorted(groups.items()):
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed for {experiment}")
        result = {"experiment": experiment, "seeds": sorted(seeds)}
        for metric in ("macro_f1", "accuracy", "weighted_f1", "clean_fpr", "hate_recall"):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            result[metric] = mean_sample_sd(values) if values else None
        output.append(result)
    return output


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"sample_id", "y_true", "y_pred"}
    if not required.issubset(left) or not required.issubset(right):
        raise ValueError("Prediction frames lack required columns")
    if left["sample_id"].duplicated().any() or right["sample_id"].duplicated().any():
        raise ValueError("Prediction frames contain duplicate IDs")
    joined = left[list(required)].merge(
        right[list(required)], on="sample_id", suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("Prediction frames do not contain identical ID sets")
    if not np.array_equal(joined["y_true_left"], joined["y_true_right"]):
        raise ValueError("Gold labels disagree after ID alignment")
    return (
        joined["y_true_left"].to_numpy(dtype=np.int64),
        joined["y_pred_left"].to_numpy(dtype=np.int64),
        joined["y_pred_right"].to_numpy(dtype=np.int64),
    )


def paired_bootstrap_delta(
    left_by_seed: Mapping[int, pd.DataFrame],
    right_by_seed: Mapping[int, pd.DataFrame],
    *,
    iterations: int = 2000,
    seed: int = 20261022,
    stratified: bool = True,
) -> dict:
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not seeds or set(left_by_seed) != set(right_by_seed):
        raise ValueError("Both experiments must have the same nonempty seed set")
    aligned = {value: _aligned(left_by_seed[value], right_by_seed[value]) for value in seeds}
    reference = aligned[seeds[0]][0]
    if any(not np.array_equal(reference, aligned[value][0]) for value in seeds[1:]):
        raise ValueError("Gold labels/ID order must agree across seeds")
    rng = np.random.default_rng(seed)
    class_indices = {label: np.flatnonzero(reference == label) for label in (0, 1, 2)}
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        if stratified:
            sample = np.concatenate([
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values() if len(indices)
            ])
        else:
            sample = rng.integers(0, len(reference), size=len(reference))
        seed_deltas = []
        for value in seeds:
            truth, left, right = aligned[value]
            left_f1 = detailed_metrics(truth[sample], left[sample])["macro_f1"]
            right_f1 = detailed_metrics(truth[sample], right[sample])["macro_f1"]
            seed_deltas.append(right_f1 - left_f1)
        deltas[iteration] = np.mean(seed_deltas)
    observed = np.mean([
        detailed_metrics(truth, right)["macro_f1"]
        - detailed_metrics(truth, left)["macro_f1"]
        for truth, left, right in aligned.values()
    ])
    return {
        "left_seeds": seeds,
        "right_seeds": seeds,
        "observed_delta_macro_f1": float(observed),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "stratified": bool(stratified),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "positive_fraction": float((deltas > 0).mean()),
    }


def main() -> None:  # pragma: no cover - reporting entry point
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    frame = pd.read_csv(args.metrics_csv)
    result = aggregate_metric_rows(frame.to_dict(orient="records"))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
