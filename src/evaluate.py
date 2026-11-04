"""Locked final-test evaluation for one trained run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .audit_data import append_test_access_log
from .checkpoints import model_artifact_hash
from .config import EXPERIMENT_MATRIX, validate_run_config
from .data import load_records_jsonl
from .io_utils import read_json, sha256_file, write_json
from .metrics import detailed_metrics, prediction_rows
from .models import load_model_artifact


def load_protocol_lock(path: str | Path) -> dict:
    lock = read_json(path)
    if not isinstance(lock, dict) or lock.get("status") != "locked":
        raise ValueError("protocol_lock.json must have status=locked")
    if lock.get("stage") != "final_evaluation":
        raise ValueError("Protocol lock is not finalized for test evaluation")
    if lock.get("protocol") != "v2":
        raise ValueError("Expected protocol v2 lock")
    for key in (
        "code_revision", "data_hash", "model_revision", "experiments", "seeds",
        "test_file_sha256", "runs",
    ):
        if not lock.get(key):
            raise ValueError(f"Protocol lock lacks {key}")
    if set(lock["experiments"]) != set(EXPERIMENT_MATRIX):
        raise ValueError("Protocol lock must contain exactly the 11 v2 experiments")
    if sorted(lock["seeds"]) != [42, 52, 62]:
        raise ValueError("Protocol lock must contain seeds 42/52/62")
    expected_runs = {
        f"{experiment}__seed{seed}__protocol_v2"
        for experiment in EXPERIMENT_MATRIX for seed in (42, 52, 62)
    }
    if set(lock["runs"]) != expected_runs:
        raise ValueError("Protocol lock must register all 33 final runs")
    return lock


def _locked_run(lock: Mapping, run_id: str) -> Mapping:
    runs = lock.get("runs")
    if not isinstance(runs, dict) or run_id not in runs:
        raise ValueError(f"Run {run_id} is not listed in protocol lock")
    return runs[run_id]


def evaluate_run(
    *,
    protocol_lock_path: str | Path,
    run_dir: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    test_access_log: str | Path,
) -> dict:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from .data import to_hf_dataset

    lock = load_protocol_lock(protocol_lock_path)
    run_dir = Path(run_dir)
    run_config = read_json(run_dir / "config.json")
    validate_run_config(run_config, execution=True)
    run_id = f"{run_config['experiment']}__seed{run_config['seed']}__protocol_{run_config['protocol']}"
    locked = _locked_run(lock, run_id)
    summary = read_json(run_dir / "run_summary.json")
    status = read_json(run_dir / "status.json")
    if status.get("state") != "completed":
        raise ValueError("Run status is not completed")
    if summary.get("checkpoint_hash") != locked.get("checkpoint_hash"):
        raise ValueError("Checkpoint hash differs from protocol lock")
    if run_config.get("code_revision") != lock.get("code_revision"):
        raise ValueError("Run code revision differs from protocol lock")
    if run_config.get("model_revision") != lock.get("model_revision"):
        raise ValueError("Run model revision differs from protocol lock")
    if run_config.get("data_hash") != lock.get("data_hash"):
        raise ValueError("Run data hash differs from protocol lock")
    if sha256_file(test_path) != lock.get("test_file_sha256"):
        raise ValueError("Test file hash differs from protocol lock")

    test_records = load_records_jsonl(test_path, expected_split="test")
    best_dir = run_dir / "best_model"
    if model_artifact_hash(best_dir) != summary.get("checkpoint_hash"):
        raise ValueError("Best-model artifact hash differs from run summary")
    tokenizer = AutoTokenizer.from_pretrained(best_dir, local_files_only=True)
    model = load_model_artifact(best_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = to_hf_dataset(test_records, tokenizer, max_length=int(run_config["max_length"]))
    collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=int(run_config["eval_batch"]), shuffle=False,
                        collate_fn=collator)
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            del labels
            chunks.append(model(**batch).logits.detach().float().cpu().numpy())
    logits = np.concatenate(chunks, axis=0)
    predicted = logits.argmax(axis=-1)
    metrics = detailed_metrics([row["labels"] for row in test_records], predicted)
    rows = prediction_rows(
        sidecar=test_records,
        logits=logits,
        experiment_id=run_config["experiment"],
        seed=int(run_config["seed"]),
        checkpoint_hash=summary["checkpoint_hash"],
        protocol_version=run_config["protocol"],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{run_id}.parquet"
    metric_path = output_dir / f"{run_id}.metrics.json"
    if prediction_path.exists() or metric_path.exists():
        raise FileExistsError(f"Refusing to overwrite final outputs for {run_id}")
    pd.DataFrame(rows).to_parquet(prediction_path, index=False)
    write_json(metric_path, metrics)
    append_test_access_log(
        test_access_log,
        phase="final_evaluation",
        code_revision=lock["code_revision"],
        purpose=f"locked final evaluation for {run_id}",
        input_hashes={
            "test_file": sha256_file(test_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "checkpoint": summary["checkpoint_hash"],
        },
        outputs=[str(prediction_path), str(metric_path)],
    )
    return metrics


def main() -> None:  # pragma: no cover - exercised in Kaggle
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-access-log", required=True)
    args = parser.parse_args()
    metrics = evaluate_run(
        protocol_lock_path=args.protocol_lock,
        run_dir=args.run_dir,
        test_path=args.test_path,
        output_dir=args.output_dir,
        test_access_log=args.test_access_log,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
