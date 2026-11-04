import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.io_utils import sha256_file
from src.qc import approve_manifest, build_qc_report


def _write_review(path, decisions):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["aug_id", "labels", "decision", "reason"]
        )
        writer.writeheader()
        for index, decision in enumerate(decisions):
            writer.writerow({
                "aug_id": f"aug_{index}", "labels": index // 2,
                "decision": decision, "reason": "",
            })


class QcTests(unittest.TestCase):
    def test_two_reviewer_adjudication_and_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "qc_status": "pending_human_review",
                "augmentation_hash": "a" * 64,
                "combined_train_hash": "b" * 64,
            }), encoding="utf-8")
            first, second, adjudication = (
                root / "first.csv", root / "second.csv", root / "adjudication.csv"
            )
            _write_review(first, ["preserved"] * 6)
            _write_review(second, ["changed"] + ["preserved"] * 5)
            with adjudication.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["aug_id", "labels", "decision", "reason"]
                )
                writer.writeheader()
                writer.writerow({
                    "aug_id": "aug_0", "labels": 0,
                    "decision": "preserved", "reason": "resolved",
                })
            report = build_qc_report(
                manifest_path=manifest,
                reviewer_1_path=first,
                reviewer_2_path=second,
                reviewer_1_id="alice",
                reviewer_2_id="bob",
                adjudication_path=adjudication,
                adjudicator_id="carol",
                systematic_error_found=False,
                required_per_class=2,
            )
            self.assertEqual(report["qc_status"], "passed")
            self.assertAlmostEqual(report["raw_agreement"], 5 / 6)
            report_path = root / "qc.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            approved_path = root / "approved.json"
            approve_manifest(manifest, report_path, approved_path)
            approved = json.loads(approved_path.read_text(encoding="utf-8"))
            self.assertEqual(approved["qc_status"], "passed")
            self.assertEqual(approved["qc_report_sha256"], sha256_file(report_path))

    def test_insufficient_samples_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "qc_status": "pending_human_review",
                "augmentation_hash": "a", "combined_train_hash": "b",
            }), encoding="utf-8")
            first, second = root / "first.csv", root / "second.csv"
            _write_review(first, ["preserved"] * 6)
            _write_review(second, ["preserved"] * 6)
            report = build_qc_report(
                manifest_path=manifest,
                reviewer_1_path=first,
                reviewer_2_path=second,
                reviewer_1_id="alice",
                reviewer_2_id="bob",
                adjudication_path=None,
                adjudicator_id=None,
                systematic_error_found=False,
                required_per_class=30,
            )
            self.assertEqual(report["qc_status"], "insufficient_samples")


if __name__ == "__main__":
    unittest.main()
