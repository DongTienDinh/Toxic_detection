"""Data-integrity audit with a narrow pre-lock test-access policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import unicodedata
from typing import Callable, Mapping, Sequence

from .data import EXPECTED_COUNTS, SPLITS, load_records_jsonl, validate_records
from .io_utils import canonical_json, sha256_file, sha256_text, write_json


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _key(text: str, *, normalized: bool) -> str:
    return sha256_text(normalized_text(text) if normalized else text)


def _duplicate_index(splits: Mapping[str, Sequence[Mapping]], *, normalized: bool) -> dict:
    index: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for split, records in splits.items():
        for record in records:
            index[_key(record["text"], normalized=normalized)].append(
                (split, record["sample_id"], record["labels"])
            )
    return {key: value for key, value in index.items() if len(value) > 1}


def _pair_counts(duplicates: Mapping[str, Sequence[tuple[str, str, int]]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for occurrences in duplicates.values():
        present = sorted({split for split, _, _ in occurrences})
        if len(present) == 1:
            counts[f"within:{present[0]}"] += 1
        else:
            for index, left in enumerate(present):
                for right in present[index + 1:]:
                    counts[f"cross:{left}-{right}"] += 1
    return dict(sorted(counts.items()))


def _label_conflicts(
    duplicates: Mapping[str, Sequence[tuple[str, str, int]]],
    *,
    allowed_splits: set[str],
) -> int:
    return sum(
        1
        for occurrences in duplicates.values()
        if {split for split, _, _ in occurrences}.issubset(allowed_splits)
        and len({label for _, _, label in occurrences}) > 1
    )


def _length_summary(records: Sequence[Mapping], token_counter: Callable[[str], int]) -> dict:
    lengths = [token_counter(record["text"]) for record in records]
    if any(type(value) is not int or value <= 0 for value in lengths):
        raise ValueError("token_counter must return positive integers")
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
        "over_128": sum(value > 128 for value in lengths),
    }


def audit_splits(
    splits: Mapping[str, Sequence[Mapping]],
    *,
    phase: str,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[dict, dict]:
    """Return (public report, private details).

    During ``integrity_audit`` test labels are only domain-validated by
    ``validate_records``. No test label histogram or label conflict is emitted.
    """
    if phase not in ("integrity_audit", "final_evaluation"):
        raise ValueError("phase must be integrity_audit or final_evaluation")
    if set(splits) != set(SPLITS):
        raise ValueError(f"Expected exactly splits {SPLITS}")
    for split in SPLITS:
        validate_records(splits[split], expected_split=split)

    exact = _duplicate_index(splits, normalized=False)
    normalized = _duplicate_index(splits, normalized=True)
    report = {
        "phase": phase,
        "schema_valid": True,
        "counts": {split: len(splits[split]) for split in SPLITS},
        "expected_counts": dict(EXPECTED_COUNTS),
        "count_checks": {
            split: len(splits[split]) == EXPECTED_COUNTS[split] for split in SPLITS
        },
        "label_distribution_train_dev": {
            split: dict(sorted(Counter(r["labels"] for r in splits[split]).items()))
            for split in ("train", "validation")
        },
        "exact_duplicate_groups": len(exact),
        "normalized_duplicate_groups": len(normalized),
        "exact_duplicate_pairs": _pair_counts(exact),
        "normalized_duplicate_pairs": _pair_counts(normalized),
        "train_dev_conflicting_label_groups": _label_conflicts(
            normalized, allowed_splits={"train", "validation"}
        ),
    }
    if token_counter is not None:
        report["token_lengths_train_dev"] = {
            split: _length_summary(splits[split], token_counter)
            for split in ("train", "validation")
        }
    if phase == "final_evaluation":
        report["label_distribution_test"] = dict(
            sorted(Counter(r["labels"] for r in splits["test"]).items())
        )
        report["all_split_conflicting_label_groups"] = _label_conflicts(
            normalized, allowed_splits=set(SPLITS)
        )

    private = {
        "phase": phase,
        "exact_duplicates": exact,
        "normalized_duplicates": normalized,
        "test_hashes": [
            {"sample_id": r["sample_id"], "text_sha256": sha256_text(r["text"])}
            for r in splits["test"]
        ],
    }
    report["report_sha256"] = sha256_text(canonical_json(report))
    return report, private


def append_test_access_log(
    path: str | Path,
    *,
    phase: str,
    code_revision: str | None,
    purpose: str,
    input_hashes: Mapping[str, str],
    outputs: Sequence[str],
) -> None:
    if phase not in ("integrity_audit", "final_evaluation"):
        raise ValueError("Invalid test access phase")
    if not purpose.strip():
        raise ValueError("Purpose is required")
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "code_revision": code_revision,
        "purpose": purpose,
        "input_hashes": dict(input_hashes),
        "outputs": list(outputs),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def save_audit(
    output_dir: str | Path,
    *,
    report: Mapping,
    private_details: Mapping,
) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    return {
        "report": write_json(target / "audit_report.json", dict(report)),
        "private": write_json(target / "audit_private.json", dict(private_details)),
    }


def main() -> None:  # pragma: no cover - integration entry point
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--phase", choices=("integrity_audit", "final_evaluation"),
        default="integrity_audit",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-access-log", required=True)
    parser.add_argument("--code-revision")
    args = parser.parse_args()
    if Path(args.output_dir).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    splits = {
        split: load_records_jsonl(path, expected_split=split)
        for split, path in paths.items()
    }
    report, private = audit_splits(splits, phase=args.phase)
    outputs = save_audit(
        args.output_dir, report=report, private_details=private
    )
    append_test_access_log(
        args.test_access_log,
        phase=args.phase,
        code_revision=args.code_revision,
        purpose=f"Explicit {args.phase} data audit",
        input_hashes={split: sha256_file(path) for split, path in paths.items()},
        outputs=[str(value) for value in outputs.values()],
    )
    # The public report deliberately omits the pre-lock test histogram.
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
