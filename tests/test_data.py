import csv
from pathlib import Path
import tempfile
import unittest

from src.audit_data import audit_splits
from src.data import (
    DatasetSchema,
    load_training_jsonl,
    read_vihsd_csv,
    save_records_jsonl,
    validate_records,
    validate_training_records,
)
from src.io_utils import write_jsonl


SCHEMA = DatasetSchema(
    text_column="free_text",
    label_column="label_id",
    source_label_to_id={"0": 0, "1": 1, "2": 2},
)


def record(split, index, text="xin chào", label=0):
    return {
        "sample_id": f"{split}_{index:08d}",
        "text": text,
        "labels": label,
        "split": split,
        "source_row": index,
    }


class DataTests(unittest.TestCase):
    def test_csv_multiline_source_row_and_extra_index(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["index", "free_text", "label_id"])
                writer.writeheader()
                writer.writerow({"index": 99, "free_text": "dòng một\ndòng hai", "label_id": 1})
                writer.writerow({"index": 2, "free_text": "bình luận", "label_id": 0})
            rows = read_vihsd_csv(path, split="train", schema=SCHEMA)
            self.assertEqual(rows[0]["source_row"], 0)
            self.assertEqual(rows[0]["sample_id"], "train_00000000")
            self.assertEqual(rows[0]["text"], "dòng một\ndòng hai")
            self.assertEqual(rows[1]["source_row"], 1)

    def test_invalid_mapping_and_empty_text_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.csv"
            path.write_text("free_text,label_id\n,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_vihsd_csv(path, split="train", schema=SCHEMA, encoding="utf-8")
            path.write_text("free_text,label_id\ntext,9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_vihsd_csv(path, split="train", schema=SCHEMA, encoding="utf-8")

    def test_augmented_train_loader_keeps_source_pointer(self):
        rows = [record("train", 0)]
        augmented = {
            **record("train", 0, "xin , chào"),
            "sample_id": "aug__aeda_vi_v1__train_00000000",
            "source_id": "train_00000000",
            "is_augmented": True,
        }
        validate_training_records([*rows, augmented])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "combined.jsonl"
            write_jsonl(path, [*rows, augmented])
            self.assertEqual(len(load_training_jsonl(path)), 2)

    def test_integrity_audit_does_not_expose_test_distribution(self):
        splits = {
            "train": [record("train", 0, "trùng", 0)],
            "validation": [record("validation", 0, "khác", 1)],
            "test": [record("test", 0, "trùng", 2)],
        }
        report, private = audit_splits(splits, phase="integrity_audit")
        self.assertNotIn("label_distribution_test", report)
        self.assertIn("test_hashes", private)
        self.assertEqual(report["normalized_duplicate_pairs"]["cross:test-train"], 1)
        final, _ = audit_splits(splits, phase="final_evaluation")
        self.assertEqual(final["label_distribution_test"], {2: 1})


if __name__ == "__main__":
    unittest.main()
