"""Strict experiment configuration loading and protocol-v2 validation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

from .io_utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER_PATTERN = re.compile(r"(?:RESOLVE|COMPUTE|LOCK|PLACEHOLDER|TODO)")

EXPERIMENT_MATRIX = {
    "BASE": ("standard", "none", None),
    "POOL": ("mean", "none", None),
    "CNN": ("cnn", "none", None),
    "CONCAT": ("concat", "none", None),
    "GATE": ("gated", "none", None),
    "BASE_AEDA": ("standard", "aeda", "aeda_vi_v1"),
    "GATE_AEDA": ("gated", "aeda", "aeda_vi_v1"),
    "GATE_TEEN": ("gated", "vietnamese", "vi_teencode_v1"),
    "GATE_ACCENT": ("gated", "vietnamese", "vi_partial_diacritics_v1"),
    "BASE_VI": ("standard", "vietnamese", "vi_social_aug_v1"),
    "GATE_VI": ("gated", "vietnamese", "vi_social_aug_v1"),
}


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on bootstrap environment
        raise RuntimeError("Install pyyaml before reading YAML config") from exc
    return yaml


def load_yaml(path: str | Path) -> dict:
    value = _yaml_module().safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_experiment_config(
    experiment: str,
    *,
    seed: int,
    base_path: str | Path = PROJECT_ROOT / "configs" / "base.yaml",
    experiments_path: str | Path = PROJECT_ROOT / "configs" / "experiments.yaml",
) -> dict:
    base = load_yaml(base_path)
    document = load_yaml(experiments_path)
    specs = document.get("experiments")
    if not isinstance(specs, dict) or experiment not in specs:
        raise ValueError(f"Unknown experiment: {experiment}")
    resolved = deep_merge(base, specs[experiment])
    resolved["experiment"] = experiment
    resolved["seed"] = seed
    validate_run_config(resolved, execution=False)
    return resolved


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")
    else:
        yield path, value


def placeholder_paths(config: Mapping[str, Any]) -> list[str]:
    return [
        path
        for path, value in _walk(config)
        if isinstance(value, str) and PLACEHOLDER_PATTERN.search(value)
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_run_config(config: Mapping[str, Any], *, execution: bool) -> None:
    _require(config.get("protocol") == "v2", "protocol must be v2")
    experiment = config.get("experiment")
    _require(experiment in EXPERIMENT_MATRIX, f"Unknown experiment: {experiment}")
    expected = EXPERIMENT_MATRIX[str(experiment)]
    actual = (
        config.get("variant"),
        config.get("augmentation"),
        config.get("augmentation_version"),
    )
    _require(actual == expected, f"Experiment matrix mismatch: expected {expected}, got {actual}")
    _require(config.get("seed") in (42, 52, 62), "seed must be one of 42/52/62")
    _require(config.get("world_size") == 1, "protocol v2 requires one logical GPU")
    _require(config.get("max_length") == 128, "protocol v2 max_length must be 128")
    _require(config.get("padding_side") == "right", "padding_side must be right")
    _require(config.get("selection_metric") == "macro_f1", "selection metric must be macro_f1")
    _require(config.get("cnn_pooling", "max") in ("max", "max_mean"),
             "cnn_pooling must be max or max_mean")
    _require(config.get("loss_type", "cross_entropy") in ("cross_entropy", "focal"),
             "loss_type must be cross_entropy or focal")
    gamma = config.get("focal_gamma", 0.0)
    _require(type(gamma) in (int, float) and gamma >= 0, "focal_gamma must be nonnegative")
    weight_power = config.get("class_weight_power", 0.5)
    _require(type(weight_power) in (int, float) and 0 <= weight_power <= 1,
             "class_weight_power must be in [0,1]")
    encoder_lr = config.get("encoder_learning_rate")
    head_lr = config.get("head_learning_rate")
    _require((encoder_lr is None) == (head_lr is None),
             "encoder/head learning rates must be set together")
    if encoder_lr is not None:
        _require(0 < float(encoder_lr) <= float(head_lr),
                 "learning rates must satisfy 0 < encoder <= head")
    patience = config.get("early_stopping_patience")
    _require(patience is None or (type(patience) is int and patience > 0),
             "early_stopping_patience must be a positive integer or null")
    rate = config.get("candidate_rate")
    _require(type(rate) in (int, float) and 0 <= rate <= 1, "candidate_rate must be in [0,1]")
    _require(type(config.get("augmentation_seed")) is int, "augmentation_seed must be int")
    label2id = config.get("label2id")
    _require(label2id == {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}, "Unexpected label mapping")

    augmented = config.get("augmentation") != "none"
    if augmented and execution:
        _require(type(config.get("matched_update_steps")) is int and config["matched_update_steps"] > 0,
                 "Augmented execution requires matched_update_steps from BASE")
        _require(type(config.get("eval_steps")) is int and config["eval_steps"] > 0,
                 "Augmented execution requires eval_steps from BASE")
        for key in ("augmentation_hash", "augmentation_manifest_path", "combined_train_hash"):
            _require(bool(config.get(key)), f"Augmented execution requires {key}")
    if config.get("augmentation") == "vietnamese" and execution:
        for key in ("lexicon_hash", "allowlist_hash", "protected_terms_hash"):
            _require(bool(config.get(key)), f"Vietnamese augmentation requires {key}")

    if execution:
        placeholders = placeholder_paths(config)
        _require(not placeholders, f"Unresolved placeholders: {placeholders}")
        for key in ("model_revision", "dataset_revision", "code_revision"):
            _require(bool(FULL_SHA_PATTERN.fullmatch(str(config.get(key, "")).lower())),
                     f"{key} must be a full 40-hex commit SHA")
        _require(bool(config.get("data_hash")), "data_hash is required")
        _require(bool(config.get("train_path")) and bool(config.get("validation_path")),
                 "train_path and validation_path are required")
        _require(not config.get("test_path"), "Training config must not expose test_path")


def run_id(config: Mapping[str, Any]) -> str:
    return f"{config['experiment']}__seed{config['seed']}__protocol_{config['protocol']}"


def write_resolved_config(path: str | Path, config: Mapping[str, Any], *, execution: bool) -> Path:
    validate_run_config(config, execution=execution)
    return write_json(path, dict(config))


def main() -> None:  # pragma: no cover - configuration entry point
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_MATRIX), required=True)
    parser.add_argument("--seed", choices=(42, 52, 62), type=int, required=True)
    parser.add_argument("--overrides", help="JSON/YAML with locked paths, hashes and revisions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution", action="store_true")
    args = parser.parse_args()
    config = resolve_experiment_config(args.experiment, seed=args.seed)
    if args.overrides:
        override_path = Path(args.overrides)
        if override_path.suffix.lower() == ".json":
            import json
            overrides = json.loads(override_path.read_text(encoding="utf-8"))
        else:
            overrides = load_yaml(override_path)
        if not isinstance(overrides, dict):
            raise ValueError("Overrides must be an object")
        config = deep_merge(config, overrides)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    write_resolved_config(target, config, execution=args.execution)
    print(target)


if __name__ == "__main__":
    main()
