import json
from pathlib import Path
import tempfile
import unittest

from src.checkpoints import validate_recovery_checkpoint, write_recovery_manifest


def _config():
    return {
        "experiment": "BASE",
        "seed": 42,
        "protocol": "v2",
        "data_hash": "d" * 64,
        "code_revision": "a" * 40,
        "model_revision": "b" * 40,
        "dataset_revision": "c" * 40,
    }


class ResumeCheckpointTests(unittest.TestCase):
    def test_complete_checkpoint_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "config.json", "model.safetensors",
            ):
                (root / name).write_bytes(b"fixture")
            (root / "trainer_state.json").write_text(
                json.dumps({"global_step": 17}), encoding="utf-8"
            )
            write_recovery_manifest(root, config=_config(), global_step=17)
            result = validate_recovery_checkpoint(root, expected_config=_config())
            self.assertEqual(result["global_step"], 17)

            changed = dict(_config(), data_hash="e" * 64)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_recovery_checkpoint(root, expected_config=changed)

    def test_weights_only_is_not_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "Not an exact recovery"):
                validate_recovery_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
