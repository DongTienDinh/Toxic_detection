"""Create a pre-training campaign lock and finalize it for test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .checkpoints import config_fingerprint, model_artifact_hash
from .config import EXPERIMENT_MATRIX, PROJECT_ROOT, run_id, validate_run_config
from .io_utils import read_json, sha256_file, write_json
from .manifests import git_revision


SEEDS = (42, 52, 62)


def _expected_ids() -> set[str]:
    return {
        f"{experiment}__seed{seed}__protocol_v2"
        for experiment in EXPERIMENT_MATRIX for seed in SEEDS
    }


def build_campaign_lock(
    *,
    config_paths: Iterable[str | Path],
    data_manifest_path: str | Path,
    test_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    require_clean_git: bool = True,
) -> dict:
    configs = []
    for path in config_paths:
        config = read_json(path)
        if not isinstance(config, dict):
            raise ValueError(f"Resolved config must be an object: {path}")
        validate_run_config(config, execution=True)
        configs.append(config)
    ids = [run_id(config) for config in configs]
    expected = _expected_ids()
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Campaign lock requires exactly 11 experiments x 3 seeds")

    singleton_fields = ("code_revision", "data_hash", "model_revision", "dataset_revision")
    shared = {}
    for key in singleton_fields:
        values = {config[key] for config in configs}
        if len(values) != 1:
            raise ValueError(f"Final configs disagree on {key}")
        shared[key] = values.pop()
    if sha256_file(data_manifest_path) != shared["data_hash"]:
        raise ValueError("data_manifest hash differs from final configs")
    data_manifest = read_json(data_manifest_path)
    if data_manifest.get("model_revision") != shared["model_revision"]:
        raise ValueError("Data manifest model revision differs from final configs")
    if data_manifest.get("dataset_revision") != shared["dataset_revision"]:
        raise ValueError("Data manifest dataset revision differs from final configs")
    expected_test_hash = data_manifest.get("prepared_jsonl_sha256", {}).get("test")
    actual_test_hash = sha256_file(test_path)
    if expected_test_hash != actual_test_hash:
        raise ValueError("Test JSONL differs from the prepared data manifest")

    if require_clean_git:
        git = git_revision(project_root)
        if git["revision"] != shared["code_revision"] or git["dirty"] is not False:
            raise ValueError("Final lock requires the matching clean Git revision")

    for config in configs:
        if config["augmentation"] == "none":
            continue
        manifest_path = Path(config["augmentation_manifest_path"])
        augmentation = read_json(manifest_path)
        if augmentation.get("qc_status") != "passed":
            raise ValueError(f"Augmentation QC is not passed: {config['experiment']}")
        if augmentation.get("augmentation_hash") != config["augmentation_hash"]:
            raise ValueError(f"Augmentation hash mismatch: {config['experiment']}")
        if augmentation.get("combined_train_hash") != config["combined_train_hash"]:
            raise ValueError(f"Combined train hash mismatch: {config['experiment']}")
        augmentation_config = augmentation.get("config", {})
        for key in ("augmentation_version", "augmentation_seed", "candidate_rate"):
            if augmentation_config.get(key) != config.get(key):
                raise ValueError(f"Augmentation {key} mismatch: {config['experiment']}")
        if augmentation.get("tokenizer_revision") != config["model_revision"]:
            raise ValueError(f"Tokenizer revision mismatch: {config['experiment']}")
        if augmentation.get("test_only_resources"):
            raise ValueError(f"TEST_ONLY augmentation is forbidden: {config['experiment']}")
        qc_path = Path(augmentation.get("qc_report_path", ""))
        if not qc_path.is_absolute():
            qc_path = manifest_path.parent / qc_path
        if (
            not qc_path.is_file()
            or sha256_file(qc_path) != augmentation.get("qc_report_sha256")
        ):
            raise ValueError(f"QC evidence mismatch: {config['experiment']}")
        qc_report = read_json(qc_path)
        if qc_report.get("qc_status") != "passed":
            raise ValueError(f"QC report is not passed: {config['experiment']}")
        for key in ("augmentation_hash", "combined_train_hash"):
            if qc_report.get(key) != augmentation.get(key):
                raise ValueError(f"QC report {key} mismatch: {config['experiment']}")
        if config["augmentation"] == "vietnamese":
            expected_resources = {
                "lexicon": config["lexicon_hash"],
                "allowlist": config["allowlist_hash"],
                "protected_terms": config["protected_terms_hash"],
            }
            if augmentation.get("resource_hashes") != expected_resources:
                raise ValueError(f"Resource hash mismatch: {config['experiment']}")

    planned = {
        run_id(config): {
            "config_sha256": config_fingerprint(config),
            "experiment": config["experiment"],
            "seed": config["seed"],
            "variant": config["variant"],
            "augmentation": config["augmentation"],
            "augmentation_version": config["augmentation_version"],
        }
        for config in configs
    }
    return {
        "schema_version": 1,
        "status": "locked",
        "stage": "final_training",
        "protocol": "v2",
        **shared,
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "test_file_sha256": actual_test_hash,
        "experiments": sorted(EXPERIMENT_MATRIX),
        "seeds": list(SEEDS),
        "planned_runs": dict(sorted(planned.items())),
    }


def finalize_evaluation_lock(
    *, campaign_lock_path: str | Path, run_root: str | Path
) -> dict:
    campaign = read_json(campaign_lock_path)
    if (
        not isinstance(campaign, dict)
        or campaign.get("status") != "locked"
        or campaign.get("stage") != "final_training"
    ):
        raise ValueError("Expected a locked final_training campaign")
    if set(campaign.get("planned_runs", {})) != _expected_ids():
        raise ValueError("Campaign does not contain all 33 planned runs")
    root = Path(run_root)
    completed = {}
    for identifier, planned in campaign["planned_runs"].items():
        run_dir = root / identifier
        config = read_json(run_dir / "config.json")
        status = read_json(run_dir / "status.json")
        summary = read_json(run_dir / "run_summary.json")
        if run_id(config) != identifier:
            raise ValueError(f"Run identity mismatch: {identifier}")
        if config_fingerprint(config) != planned["config_sha256"]:
            raise ValueError(f"Run config changed after lock: {identifier}")
        if status.get("state") != "completed":
            raise ValueError(f"Run is not completed: {identifier}")
        actual_hash = model_artifact_hash(run_dir / "best_model")
        if summary.get("checkpoint_hash") != actual_hash:
            raise ValueError(f"Checkpoint hash mismatch: {identifier}")
        completed[identifier] = {
            **planned,
            "checkpoint_hash": actual_hash,
            "global_step": summary.get("global_step"),
            "best_metric": summary.get("best_metric"),
        }
    result = {
        key: value for key, value in campaign.items() if key != "planned_runs"
    }
    result["stage"] = "final_evaluation"
    result["campaign_lock_sha256"] = sha256_file(campaign_lock_path)
    result["runs"] = dict(sorted(completed.items()))
    return result


def main() -> None:  # pragma: no cover - campaign entry point
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--config-dir", required=True)
    campaign.add_argument("--data-manifest", required=True)
    campaign.add_argument("--test-path", required=True)
    campaign.add_argument("--project-root", default=str(PROJECT_ROOT))
    campaign.add_argument("--output", required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--campaign-lock", required=True)
    final.add_argument("--run-root", required=True)
    final.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.command == "campaign":
        configs = sorted(Path(args.config_dir).glob("*.json"))
        lock = build_campaign_lock(
            config_paths=configs,
            data_manifest_path=args.data_manifest,
            test_path=args.test_path,
            project_root=args.project_root,
        )
    else:
        lock = finalize_evaluation_lock(
            campaign_lock_path=args.campaign_lock, run_root=args.run_root
        )
    write_json(args.output, lock)
    print(json.dumps({
        "output": args.output, "stage": lock["stage"],
        "runs": len(lock.get("runs", lock.get("planned_runs", {}))),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
