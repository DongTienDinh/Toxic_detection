"""Single-experiment Trainer harness for protocol v2.

The training runner intentionally accepts train/dev only. Test data belongs to
``src.evaluate`` and is rejected by config validation here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Mapping

from .config import PROJECT_ROOT, run_id, validate_run_config
from .checkpoints import (
    config_fingerprint, model_artifact_hash, validate_recovery_checkpoint,
    write_recovery_manifest,
)
from .data import (
    load_records_jsonl, load_training_jsonl, records_hash, to_hf_dataset,
    training_records_hash,
)
from .io_utils import read_json, sha256_file, write_json
from .manifests import write_environment_manifest


def load_config_file(path: str | Path) -> dict:
    target = Path(path)
    if target.suffix.lower() == ".json":
        value = read_json(target)
    else:
        from .config import load_yaml
        value = load_yaml(target)
    if not isinstance(value, dict):
        raise ValueError("Run config must be an object")
    return value


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import set_seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _verify_manifest(config: Mapping) -> dict:
    path = config.get("data_manifest_path")
    if not path:
        raise ValueError("data_manifest_path is required")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256_file(manifest_path) != config["data_hash"]:
        raise ValueError("data_hash does not match data_manifest_path")
    manifest = read_json(manifest_path)
    if manifest.get("model_revision") != config["model_revision"]:
        raise ValueError("Model revision differs from data manifest")
    if manifest.get("dataset_revision") != config["dataset_revision"]:
        raise ValueError("Dataset revision differs from data manifest")
    if config["augmentation"] != "none":
        aug_path = Path(config["augmentation_manifest_path"])
        augmentation = read_json(aug_path)
        if augmentation.get("combined_train_hash") != config["combined_train_hash"]:
            raise ValueError("combined_train_hash mismatch")
        if augmentation.get("augmentation_hash") != config["augmentation_hash"]:
            raise ValueError("augmentation_hash mismatch")
        if augmentation.get("qc_status") != "passed":
            raise ValueError("Augmentation human QC has not passed")
        qc_path = augmentation.get("qc_report_path")
        qc_hash = augmentation.get("qc_report_sha256")
        if qc_path:
            qc_path = Path(qc_path)
            if not qc_path.is_absolute():
                qc_path = aug_path.parent / qc_path
        if not qc_path or not qc_hash or not qc_path.is_file():
            raise ValueError("Approved augmentation manifest lacks a readable QC report")
        if sha256_file(qc_path) != qc_hash:
            raise ValueError("Augmentation QC report hash mismatch")
        qc_report = read_json(qc_path)
        if qc_report.get("qc_status") != "passed":
            raise ValueError("Augmentation QC report is not passed")
        for key in ("augmentation_hash", "combined_train_hash"):
            if qc_report.get(key) != augmentation.get(key):
                raise ValueError(f"Augmentation QC report {key} mismatch")
        if augmentation.get("test_only_resources"):
            raise ValueError("TEST_ONLY resources cannot be used for training")
        augmentation_config = augmentation.get("config", {})
        if augmentation_config.get("augmentation_version") != config["augmentation_version"]:
            raise ValueError("Augmentation version mismatch")
        if augmentation_config.get("augmentation_seed") != config["augmentation_seed"]:
            raise ValueError("Augmentation seed mismatch")
        if augmentation_config.get("candidate_rate") != config["candidate_rate"]:
            raise ValueError("Augmentation candidate rate mismatch")
        if augmentation.get("tokenizer_revision") != config["model_revision"]:
            raise ValueError("Augmentation tokenizer revision differs from model revision")
        if config["augmentation"] == "vietnamese":
            resource_hashes = augmentation.get("resource_hashes", {})
            expected = {
                "lexicon": config["lexicon_hash"],
                "allowlist": config["allowlist_hash"],
                "protected_terms": config["protected_terms_hash"],
            }
            if resource_hashes != expected:
                raise ValueError("Vietnamese resource hashes differ from run config")
    return manifest


def _training_arguments(config: Mapping, run_dir: Path):
    from transformers import TrainingArguments
    augmented = config["augmentation"] != "none"
    kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=float(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["micro_batch"]),
        per_device_eval_batch_size=int(config["eval_batch"]),
        gradient_accumulation_steps=int(config["gradient_accumulation"]),
        optim="adamw_torch",
        adam_beta1=float(config["adam_beta1"]),
        adam_beta2=float(config["adam_beta2"]),
        adam_epsilon=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
        lr_scheduler_type="linear",
        warmup_ratio=float(config["warmup_ratio"]),
        max_grad_norm=float(config["max_grad_norm"]),
        fp16=config["precision"] == "fp16",
        bf16=config["precision"] == "bf16",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=int(config["save_total_limit"]),
        save_safetensors=True,
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        label_names=["labels"],
        save_only_model=False,
    )
    if augmented:
        kwargs.update(
            max_steps=int(config["matched_update_steps"]),
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=int(config["eval_steps"]),
            save_steps=int(config["eval_steps"]),
        )
    else:
        kwargs.update(eval_strategy="epoch", save_strategy="epoch")
    return TrainingArguments(**kwargs)


def _differential_optimizer(model, config: Mapping):
    """Use a conservative encoder LR and a faster randomly initialized head LR."""
    encoder_lr = config.get("encoder_learning_rate")
    head_lr = config.get("head_learning_rate")
    if encoder_lr is None and head_lr is None:
        return None
    if encoder_lr is None or head_lr is None:
        raise ValueError("encoder_learning_rate and head_learning_rate must be set together")
    import torch

    groups = []
    weight_decay = float(config["weight_decay"])
    for is_encoder, learning_rate in ((True, float(encoder_lr)), (False, float(head_lr))):
        for use_decay in (True, False):
            parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and name.startswith("encoder.") == is_encoder
                and (parameter.ndim > 1) == use_decay
            ]
            if parameters:
                groups.append({
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay if use_decay else 0.0,
                })
    if not groups:
        raise ValueError("No trainable parameters found for optimizer")
    return torch.optim.AdamW(
        groups,
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_epsilon"]),
    )


def _write_status(path: Path, state: str, **extra) -> None:
    write_json(path, {
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    })


def run_training(
    config: Mapping,
    *,
    token: str | None,
    output_root: str | Path,
    resume_from_checkpoint: str | None = None,
) -> dict:
    validate_run_config(config, execution=True)
    if int(config["world_size"]) != 1:
        raise ValueError("Only one logical GPU is permitted")
    data_manifest = _verify_manifest(config)
    if resume_from_checkpoint is not None:
        validate_recovery_checkpoint(
            resume_from_checkpoint, expected_config=config
        )
    set_reproducibility(int(config["seed"]))

    import numpy as np
    import pandas as pd
    import torch
    from transformers import (
        AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
        TrainerCallback,
    )
    from .metrics import compute_metrics, detailed_metrics, prediction_rows
    from .models import balanced_class_weights, build_model

    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires a CUDA GPU; use unit tests for CPU validation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one logical GPU before importing torch")

    identifier = run_id(config)
    run_dir = Path(output_root) / identifier
    if run_dir.exists() and resume_from_checkpoint is None:
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    if run_dir.exists() and resume_from_checkpoint is not None:
        existing_config_path = run_dir / "config.json"
        if not existing_config_path.is_file():
            raise ValueError("Existing resume run directory lacks config.json")
        existing_config = read_json(existing_config_path)
        if config_fingerprint(existing_config) != config_fingerprint(config):
            raise ValueError("Existing run directory config differs from resume config")
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    _write_status(status_path, "running", run_id=identifier)
    write_json(run_dir / "config.json", dict(config))
    write_environment_manifest(run_dir / "environment.json", PROJECT_ROOT)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=config["model_revision"], token=token
        )
        tokenizer.padding_side = "right"
        train_records = load_training_jsonl(config["train_path"])
        dev_records = load_records_jsonl(
            config["validation_path"], expected_split="validation"
        )
        actual_train_hash = training_records_hash(train_records)
        expected_train_hash = (
            config["combined_train_hash"]
            if config["augmentation"] != "none"
            else data_manifest["splits"]["train"]["records_sha256"]
        )
        if actual_train_hash != expected_train_hash:
            raise ValueError("Training records do not match the locked manifest hash")
        if records_hash(dev_records) != data_manifest["splits"]["validation"]["records_sha256"]:
            raise ValueError("Validation records do not match the locked data manifest")
        train_dataset = to_hf_dataset(
            train_records, tokenizer, max_length=int(config["max_length"]),
            allow_augmented_train=True,
        )
        dev_dataset = to_hf_dataset(
            dev_records, tokenizer, max_length=int(config["max_length"])
        )
        collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
        )
        class_counts = {
            str(label): sum(int(row["labels"]) == label for row in train_records)
            for label in range(3)
        }
        class_weights = None
        if config.get("loss_type", "cross_entropy") == "focal":
            class_weights = balanced_class_weights(
                [int(row["labels"]) for row in train_records],
                power=float(config.get("class_weight_power", 0.5)),
            )
        write_json(run_dir / "training_distribution.json", {
            "source": "training_split_only",
            "class_counts": class_counts,
            "class_weights": class_weights,
        })
        model, loading_info = build_model(
            config, token=token, class_weights=class_weights
        )
        write_json(run_dir / "loading_info.json", {
            key: value for key, value in loading_info.items()
            if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
        })
        args = _training_arguments(config, run_dir)
        class RecoveryManifestCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                write_recovery_manifest(
                    checkpoint, config=config, global_step=int(state.global_step)
                )
                return control

        callbacks = [RecoveryManifestCallback()]
        patience = config.get("early_stopping_patience")
        if patience is not None:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=int(patience),
                early_stopping_threshold=0.001,
            ))
        optimizer = _differential_optimizer(model, config)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=(optimizer, None),
        )
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        best_dir = run_dir / "best_model"
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)
        predictions = trainer.predict(dev_dataset)
        logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        predicted = np.asarray(logits).argmax(axis=-1)
        metrics = detailed_metrics([row["labels"] for row in dev_records], predicted)
        checkpoint_hash = model_artifact_hash(best_dir)
        rows = prediction_rows(
            sidecar=dev_records,
            logits=np.asarray(logits),
            experiment_id=str(config["experiment"]),
            seed=int(config["seed"]),
            checkpoint_hash=checkpoint_hash,
            protocol_version=str(config["protocol"]),
        )
        pd.DataFrame(rows).to_parquet(run_dir / "predictions_dev.parquet", index=False)
        write_json(run_dir / "dev_metrics.json", metrics)
        trainer.save_state()
        trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
        summary = {
            "run_id": identifier,
            "global_step": int(trainer.state.global_step),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "train_metrics": result.metrics,
            "dev_metrics": metrics,
            "checkpoint_hash": checkpoint_hash,
        }
        write_json(run_dir / "run_summary.json", summary)
        _write_status(status_path, "completed", run_id=identifier)
        return summary
    except KeyboardInterrupt as exc:
        _write_status(
            status_path,
            "interrupted",
            run_id=identifier,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    except Exception as exc:
        _write_status(
            status_path,
            "failed",
            run_id=identifier,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


def main() -> None:  # pragma: no cover - exercised in Kaggle
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = load_config_file(args.config)
    token = os.environ.get("HF_TOKEN")
    result = run_training(
        config,
        token=token,
        output_root=args.output_root,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
