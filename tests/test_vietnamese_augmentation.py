import json
from pathlib import Path
import tempfile
import unittest
import unicodedata

from src.prepare_augmentation import export_vietnamese_bundle
from src.vietnamese_augmentation import (
    VietnameseAugmentationConfig,
    build_vietnamese_augmented_train,
    find_diacritic_matches,
    find_teencode_matches,
    protected_ranges,
    remove_vietnamese_diacritics,
    scan_units,
)


def approved_resource(resource_type, entries):
    return {
        "resource_type": resource_type,
        "version": "test-v1",
        "status": "locked",
        "source_resource": "TEST_ONLY",
        "source_revision": "0" * 40,
        "source_license": "TEST_ONLY",
        "entries": entries,
    }


LEXICON = approved_resource("teencode_lexicon", [
    {
        "entry_id": "teen-ban",
        "source_text": "bạn",
        "variants": ["bn", "b"],
        "review_status": "approved",
        "reviewers": ["tester-a", "tester-b"],
    },
    {
        "entry_id": "teen-noi-chuyen",
        "source_text": "nói chuyện",
        "variants": ["nc"],
        "review_status": "approved",
        "reviewers": ["tester-a", "tester-b"],
    },
])
ALLOWLIST = approved_resource("diacritics_allowlist", [
    {
        "entry_id": "accent-chuyen",
        "source_text": "chuyện",
        "review_status": "approved",
        "reviewers": ["tester-a", "tester-b"],
    },
])
PROTECTED = approved_resource("protected_terms", [
    {
        "entry_id": "protect-khong",
        "term": "không",
        "review_status": "approved",
        "reviewers": ["tester-a", "tester-b"],
    },
])


def row(index, text, label=0):
    return {
        "sample_id": f"train_{index:08d}",
        "source_row": index,
        "split": "train",
        "text": text,
        "labels": label,
    }


def counter(text):
    return len(text.split()) + 2


def build(rows, recipe="vi_social_aug_v1"):
    return build_vietnamese_augmented_train(
        rows,
        counter,
        VietnameseAugmentationConfig(
            augmentation_version=recipe,
            candidate_rate=1,
        ),
        lexicon=LEXICON,
        allowlist=ALLOWLIST,
        protected_terms=PROTECTED,
        tokenizer_id="TEST_ONLY_FAKE",
        tokenizer_revision="0" * 40,
        require_locked_resources=True,
    )


class VietnameseAugmentationTests(unittest.TestCase):
    def test_unicode_scanner_keeps_combining_marks(self):
        nfd = unicodedata.normalize("NFD", "chuyện")
        units = scan_units(f"xin {nfd} nhé")
        self.assertEqual(len(units), 3)
        self.assertEqual(unicodedata.normalize("NFC", units[1].text), "chuyện")
        self.assertEqual(remove_vietnamese_diacritics("Đặng"), "Dang")

    def test_longest_overlap_and_protected_ranges(self):
        text = "bạn nói chuyện không @ban https://x.vn"
        units = scan_units(text)
        protected = protected_ranges(text, units, PROTECTED, require_locked=True)
        matches = find_teencode_matches(
            text, units, LEXICON, protected, require_locked=True
        )
        self.assertEqual([(m.before, m.entry_id) for m in matches], [
            ("bạn", "teen-ban"), ("nói chuyện", "teen-noi-chuyen")
        ])
        self.assertTrue(any(text[start:end] == "không" for start, end in protected))
        self.assertTrue(any(text[start:end] == "@ban" for start, end in protected))

    def test_diacritic_allowlist_and_protection(self):
        text = "chuyện không"
        units = scan_units(text)
        protected = protected_ranges(text, units, PROTECTED, require_locked=True)
        matches = find_diacritic_matches(
            text, units, ALLOWLIST, protected, require_locked=True
        )
        self.assertEqual(matches[0].variants, ("chuyen",))

    def test_same_candidate_pool_for_all_recipes(self):
        rows = [row(i, "bạn nói chuyện", i % 3) for i in range(12)]
        candidate_sets = []
        for recipe in (
            "vi_teencode_v1", "vi_partial_diacritics_v1", "vi_social_aug_v1"
        ):
            result = build(rows, recipe)
            candidate_sets.append({item["source_id"] for item in result["provenance"]})
        self.assertEqual(candidate_sets[0], candidate_sets[1])
        self.assertEqual(candidate_sets[1], candidate_sets[2])

    def test_deterministic_and_one_edit(self):
        rows = [row(i, "bạn kể chuyện", i % 3) for i in range(12)]
        first = build(rows)
        second = build(list(reversed(rows)))
        self.assertEqual(first, second)
        for provenance in first["provenance"]:
            if provenance["accepted"]:
                self.assertIn(provenance["selected_operation"], {"teencode", "diacritics"})
                self.assertIsNotNone(provenance["entry_id"])

    def test_unlocked_resource_is_rejected(self):
        resource = dict(LEXICON)
        resource["status"] = "unverified"
        with self.assertRaisesRegex(ValueError, "not approved/locked"):
            build_vietnamese_augmented_train(
                [row(0, "bạn kể chuyện")], counter,
                VietnameseAugmentationConfig(candidate_rate=1),
                lexicon=resource, allowlist=ALLOWLIST, protected_terms=PROTECTED,
                tokenizer_id="x", tokenizer_revision="0" * 40,
            )

    def test_export_roundtrip_and_qc_pending(self):
        result = build([row(i, "bạn kể chuyện", i % 3) for i in range(12)])
        self.assertEqual(result["manifest"]["qc_status"], "pending_human_review")
        with tempfile.TemporaryDirectory() as root:
            target = export_vietnamese_bundle(result, Path(root) / "bundle")
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("combined_train_hash", manifest)
            with self.assertRaises(FileExistsError):
                export_vietnamese_bundle(result, target)


if __name__ == "__main__":
    unittest.main()
