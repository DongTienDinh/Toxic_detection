"""CPU contract tests using a fake counter, not a ViSoBERT tokenizer benchmark."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.augmentation import (
    AugmentationConfig, PUNCT, build_augmented_train, export_bundle,
    generate_variant, hf_token_counter, normalized_key,
)


def row(i, text=None, label=0):
    return {"sample_id": f"train_{i:08d}", "source_row": i, "split": "train",
            "text": text if text is not None else f"bình luận số {i}", "labels": label}


def fake_counter(text):
    # Only for structural tests. Production requires the locked HF tokenizer.
    return len(text.split()) + 2


def build(rows, cfg=None, counter=fake_counter):
    return build_augmented_train(rows, counter, cfg or AugmentationConfig(),
                                 tokenizer_id="TEST_ONLY_FAKE",
                                 tokenizer_revision="0" * 40)


class AugmentationTests(unittest.TestCase):
    def test_floor_per_class_and_no_replacement(self):
        rows = [row(c * 100 + i, "một", c) for c in range(3) for i in range(5)]
        result = build(rows)
        self.assertEqual(result["manifest"]["n_candidates"], 3)
        self.assertEqual(result["manifest"]["n_accepted"], 0)
        self.assertEqual(len(result["train_combined"]), 15)
        self.assertEqual(result["manifest"]["rejections"], {"too_few_units": 3})

    def test_deterministic_despite_order_and_global_rng(self):
        rows = [row(i, label=i % 3) for i in range(48)]
        expected = build(rows)
        random.Random(200).shuffle(rows)
        random.seed(900)
        self.assertEqual(expected, build(rows))

    def test_selected_gap_whitespace(self):
        cfg = AugmentationConfig()
        for source in ("xin chào", "xin   chào", "xin\tchào", "  xin\nchào  "):
            actual, ops = generate_variant(source, "train_00000000", cfg)
            self.assertEqual(len(ops), 1)
            self.assertIn(actual.strip(), [f"xin {p} chào" for p in PUNCT])
            self.assertEqual(actual.startswith("  "), source.startswith("  "))
            self.assertEqual(actual.endswith("  "), source.endswith("  "))

    def test_multiple_insertions_preserve_word_sequence(self):
        words = [f"w{i}" for i in range(80)]
        out, ops = generate_variant(" ".join(words), "train_00000003", AugmentationConfig())
        self.assertEqual([w for w in out.split() if w not in PUNCT], words)
        self.assertEqual(len(out.split()), len(words) + len(ops))
        self.assertTrue(1 <= len(ops) <= 8)

    def test_length_boundary_and_rejection_precedence(self):
        cfg = AugmentationConfig(candidate_rate=1, max_length=6)
        rows = [row(0, "a b c d e"), row(1, "a b c d"), row(2, "a b c")]
        result = build(rows, cfg)
        reasons = [p["rejection_reason"] for p in result["provenance"]]
        self.assertEqual(reasons, ["original_too_long", "augmented_too_long", None])
        self.assertEqual(result["provenance"][2]["augmented_length"], 6)
        self.assertEqual(len(result["train_original"]), 3)

    def test_duplicate_against_original_uses_normalized_key(self):
        cfg = AugmentationConfig(candidate_rate=1)
        variant, _ = generate_variant("xin chào", "train_00000000", cfg)
        result = build([row(0, "xin chào"), row(1, "  " + variant + "  ")], cfg)
        self.assertEqual(result["provenance"][0]["rejection_reason"], "duplicate")
        self.assertEqual(normalized_key("a\t b"), normalized_key("a b"))
        self.assertNotEqual(normalized_key("A b"), normalized_key("a b"))

    def test_duplicate_among_augmented_keeps_first_source(self):
        cfg = AugmentationConfig(candidate_rate=1)
        # Seven identical two-unit inputs and six punctuation choices force a collision.
        result = build([row(i, "xin chào") for i in range(7)], cfg)
        self.assertGreaterEqual(result["manifest"]["rejections"].get("duplicate", 0), 1)
        keys = [normalized_key(r["text"]) for r in result["train_augmented"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_reject_invalid_input_and_non_train(self):
        for change in ({"split": "test"}, {"labels": "0"}, {"source_row": -1},
                       {"text": None}, {"text": " "}, {"sample_id": "x"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build([{**row(0), **change}])
        with self.assertRaises(ValueError):
            build([row(0), row(0)])

    def test_config_and_revision_validation(self):
        for change in ({"candidate_rate": float("nan")}, {"candidate_rate": 1.1},
                       {"augmentation_seed": True}, {"augmentation_version": "unknown"}):
            with self.assertRaises(ValueError):
                AugmentationConfig(**change)
        with self.assertRaises(ValueError):
            build_augmented_train([row(0)], fake_counter, AugmentationConfig(),
                                  tokenizer_id="x", tokenizer_revision="main")

    def test_labels_original_text_and_corpus_hash(self):
        rows = [row(i, label=i % 3) for i in range(12)]
        result = build(rows, AugmentationConfig(candidate_rate=1))
        lookup = {r["sample_id"]: r for r in rows}
        for r in result["train_augmented"]:
            self.assertEqual(r["labels"], lookup[r["source_id"]]["labels"])
        self.assertEqual([r["text"] for r in result["train_original"]],
                         [r["text"] for r in rows])
        changed = build([row(0, "văn bản đã đổi"), *rows[1:]], AugmentationConfig(candidate_rate=1))
        self.assertNotEqual(result["manifest"]["combined_train_hash"],
                            changed["manifest"]["combined_train_hash"])

    def test_tokenizer_adapter_counts_untruncated_special_tokens(self):
        calls = []
        def fake_tokenizer(text, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [0, 4, 5, 6, 2]}
        self.assertEqual(hf_token_counter(fake_tokenizer)("xin chào"), 5)
        self.assertTrue(calls[0]["add_special_tokens"])
        self.assertFalse(calls[0]["truncation"])

    def test_export_roundtrip_and_no_overwrite(self):
        result = build([row(i, label=i % 3) for i in range(12)])
        self.assertEqual(result["manifest"]["qc_status"], "pending_human_review")
        self.assertFalse(result["manifest"]["audit_sample_complete"])
        with tempfile.TemporaryDirectory() as root:
            target = export_bundle(result, Path(root) / "bundle")
            data = [json.loads(s) for s in (target / "train_combined.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(data, result["train_combined"])
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["output_file_sha256"]), 5)
            with self.assertRaises(FileExistsError):
                export_bundle(result, target)


if __name__ == "__main__":
    unittest.main()
