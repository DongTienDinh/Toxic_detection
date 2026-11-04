import copy
import unittest

from src.config import EXPERIMENT_MATRIX, resolve_experiment_config, validate_run_config


class ConfigTests(unittest.TestCase):
    def test_all_experiments_resolve_to_locked_matrix(self):
        for experiment, expected in EXPERIMENT_MATRIX.items():
            with self.subTest(experiment=experiment):
                config = resolve_experiment_config(experiment, seed=42)
                self.assertEqual(
                    (config["variant"], config["augmentation"], config["augmentation_version"]),
                    expected,
                )

    def test_matrix_mismatch_is_rejected(self):
        config = resolve_experiment_config("GATE_VI", seed=42)
        config["variant"] = "standard"
        with self.assertRaises(ValueError):
            validate_run_config(config, execution=False)

    def test_gated_runs_use_full_epoch_budget(self):
        gated = [name for name, spec in EXPERIMENT_MATRIX.items() if spec[0] == "gated"]
        self.assertTrue(gated)
        for experiment in gated:
            with self.subTest(experiment=experiment):
                config = resolve_experiment_config(experiment, seed=42)
                self.assertIsNone(config["early_stopping_patience"])

    def test_execution_rejects_placeholders_and_test_path(self):
        config = resolve_experiment_config("BASE", seed=42)
        with self.assertRaises(ValueError):
            validate_run_config(config, execution=True)
        config.update({
            "model_revision": "a" * 40,
            "dataset_revision": "b" * 40,
            "code_revision": "c" * 40,
            "data_hash": "d" * 64,
            "train_path": "train.jsonl",
            "validation_path": "dev.jsonl",
            "test_path": "test.jsonl",
        })
        with self.assertRaisesRegex(ValueError, "test_path"):
            validate_run_config(config, execution=True)

    def test_augmented_execution_requires_budget_and_hashes(self):
        config = resolve_experiment_config("GATE_AEDA", seed=52)
        config.update({
            "model_revision": "a" * 40,
            "dataset_revision": "b" * 40,
            "code_revision": "c" * 40,
            "data_hash": "d" * 64,
            "train_path": "train.jsonl",
            "validation_path": "dev.jsonl",
        })
        with self.assertRaisesRegex(ValueError, "matched_update_steps"):
            validate_run_config(config, execution=True)


if __name__ == "__main__":
    unittest.main()
