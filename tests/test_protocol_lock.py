from pathlib import Path
import tempfile
import unittest

from src.checkpoints import model_artifact_hash
from src.config import EXPERIMENT_MATRIX, resolve_experiment_config, run_id
from src.io_utils import sha256_file, write_json
from src.protocol_lock import build_campaign_lock, finalize_evaluation_lock


class ProtocolLockTests(unittest.TestCase):
    def test_full_campaign_and_finalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_path = root / "test.jsonl"
            test_path.write_text('{"fixture":true}\n', encoding="utf-8")
            model_revision = "b" * 40
            dataset_revision = "c" * 40
            code_revision = "a" * 40
            data_manifest_path = root / "data_manifest.json"
            write_json(data_manifest_path, {
                "model_revision": model_revision,
                "dataset_revision": dataset_revision,
                "prepared_jsonl_sha256": {"test": sha256_file(test_path)},
            })
            data_hash = sha256_file(data_manifest_path)
            qc_path = root / "qc.json"
            write_json(qc_path, {
                "qc_status": "passed",
                "augmentation_hash": "d" * 64,
                "combined_train_hash": "e" * 64,
            })

            manifests = {}
            for version in (
                "aeda_vi_v1", "vi_teencode_v1", "vi_partial_diacritics_v1",
                "vi_social_aug_v1",
            ):
                path = root / f"{version}.approved.json"
                write_json(path, {
                    "config": {
                        "augmentation_version": version,
                        "augmentation_seed": 20261022,
                        "candidate_rate": 0.25,
                    },
                    "tokenizer_revision": model_revision,
                    "augmentation_hash": "d" * 64,
                    "combined_train_hash": "e" * 64,
                    "qc_status": "passed",
                    "qc_report_path": qc_path.name,
                    "qc_report_sha256": sha256_file(qc_path),
                    "test_only_resources": False,
                    "resource_hashes": {
                        "lexicon": "f" * 64,
                        "allowlist": "1" * 64,
                        "protected_terms": "2" * 64,
                    },
                })
                manifests[version] = path

            config_dir = root / "configs"
            config_dir.mkdir()
            config_paths = []
            configs = {}
            for experiment in EXPERIMENT_MATRIX:
                for seed in (42, 52, 62):
                    config = resolve_experiment_config(experiment, seed=seed)
                    config.update({
                        "model_revision": model_revision,
                        "dataset_revision": dataset_revision,
                        "code_revision": code_revision,
                        "data_hash": data_hash,
                        "data_manifest_path": str(data_manifest_path),
                        "train_path": str(root / "train.jsonl"),
                        "validation_path": str(root / "validation.jsonl"),
                        "test_path": None,
                    })
                    if config["augmentation"] != "none":
                        config.update({
                            "matched_update_steps": 10,
                            "eval_steps": 2,
                            "augmentation_hash": "d" * 64,
                            "combined_train_hash": "e" * 64,
                            "augmentation_manifest_path": str(
                                manifests[config["augmentation_version"]]
                            ),
                        })
                    if config["augmentation"] == "vietnamese":
                        config.update({
                            "lexicon_hash": "f" * 64,
                            "allowlist_hash": "1" * 64,
                            "protected_terms_hash": "2" * 64,
                        })
                    identifier = run_id(config)
                    path = config_dir / f"{identifier}.json"
                    write_json(path, config)
                    config_paths.append(path)
                    configs[identifier] = config

            campaign = build_campaign_lock(
                config_paths=config_paths,
                data_manifest_path=data_manifest_path,
                test_path=test_path,
                require_clean_git=False,
            )
            self.assertEqual(len(campaign["planned_runs"]), 33)
            campaign_path = root / "campaign.json"
            write_json(campaign_path, campaign)

            run_root = root / "runs"
            for identifier, config in configs.items():
                run_dir = run_root / identifier
                best = run_dir / "best_model"
                best.mkdir(parents=True)
                write_json(run_dir / "config.json", config)
                write_json(run_dir / "status.json", {"state": "completed"})
                write_json(best / "config.json", {"model_type": "fixture"})
                (best / "model.safetensors").write_bytes(b"fixture")
                write_json(run_dir / "run_summary.json", {
                    "checkpoint_hash": model_artifact_hash(best),
                    "global_step": 10,
                    "best_metric": 0.5,
                })
            final = finalize_evaluation_lock(
                campaign_lock_path=campaign_path, run_root=run_root
            )
            self.assertEqual(final["stage"], "final_evaluation")
            self.assertEqual(len(final["runs"]), 33)


if __name__ == "__main__":
    unittest.main()
