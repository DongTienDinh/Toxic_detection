"""Validation and provenance for exact Trainer recovery checkpoints.

A best-model artifact is sufficient for inference, but it is not an exact
training resume.  This module deliberately distinguishes the two concepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .config import run_id
from .io_utils import canonical_json, read_json, sha256_file, sha256_text, write_json


STATE_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "config.json",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def model_artifact_hash(model_dir: str | Path) -> str:
    """Hash every file in an offline inference artifact deterministically."""
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files or not any(
        path.name in WEIGHT_FILES or path.name.endswith(".safetensors")
        for path in files
    ):
        raise ValueError("No model weight files found in inference artifact")
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return sha256_text(canonical_json(inventory))


def config_fingerprint(config: Mapping) -> str:
    """Hash the complete resolved run configuration."""
    return sha256_text(canonical_json(dict(config)))


def write_recovery_manifest(
    checkpoint_dir: str | Path,
    *,
    config: Mapping,
    global_step: int,
) -> Path:
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("global_step must be a positive integer")
    return write_json(Path(checkpoint_dir) / "recovery_manifest.json", {
        "schema_version": 1,
        "run_id": run_id(config),
        "global_step": global_step,
        "config_sha256": config_fingerprint(config),
        "data_hash": config.get("data_hash"),
        "code_revision": config.get("code_revision"),
        "model_revision": config.get("model_revision"),
        "dataset_revision": config.get("dataset_revision"),
    })


def validate_recovery_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_config: Mapping | None = None,
) -> dict:
    """Fail closed unless a checkpoint contains full recovery state.

    Runtime equivalence (continuous run versus interrupted/resumed run) still
    has to be demonstrated in the target Kaggle GPU environment.  This check
    establishes file completeness and provenance before Trainer reads state.
    """
    target = Path(checkpoint_dir)
    if not target.is_dir():
        raise FileNotFoundError(target)
    missing = [name for name in STATE_FILES if not (target / name).is_file()]
    if not any((target / name).is_file() for name in WEIGHT_FILES):
        missing.append("one of model.safetensors/pytorch_model.bin")
    manifest_path = target / "recovery_manifest.json"
    if not manifest_path.is_file():
        missing.append("recovery_manifest.json")
    if missing:
        raise ValueError(f"Not an exact recovery checkpoint; missing: {missing}")

    state = read_json(target / "trainer_state.json")
    manifest = read_json(manifest_path)
    if not isinstance(state, dict) or not isinstance(manifest, dict):
        raise ValueError("Checkpoint state/manifest must be JSON objects")
    state_step = state.get("global_step")
    manifest_step = manifest.get("global_step")
    if type(state_step) is not int or state_step <= 0 or state_step != manifest_step:
        raise ValueError("Recovery manifest global_step differs from Trainer state")
    if expected_config is not None:
        expected = {
            "run_id": run_id(expected_config),
            "config_sha256": config_fingerprint(expected_config),
            "data_hash": expected_config.get("data_hash"),
            "code_revision": expected_config.get("code_revision"),
            "model_revision": expected_config.get("model_revision"),
            "dataset_revision": expected_config.get("dataset_revision"),
        }
        mismatches = {
            key: {"expected": value, "found": manifest.get(key)}
            for key, value in expected.items() if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Recovery checkpoint provenance mismatch: {mismatches}")
    return {
        "checkpoint": str(target.resolve()),
        "global_step": state_step,
        "has_scaler_state": (target / "scaler.pt").is_file(),
        "manifest": manifest,
    }
