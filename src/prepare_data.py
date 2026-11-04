"""Prepare immutable, audited ViHSD JSONL artifacts from verified CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping

from .audit_data import append_test_access_log, audit_splits, save_audit
from .data import (
    EXPECTED_COUNTS,
    SPLITS,
    DatasetSchema,
    build_data_manifest,
    read_vihsd_csv,
    save_records_jsonl,
)
from .io_utils import sha256_file, write_json


def prepare_csv_bundle(
    *,
    csv_paths: Mapping[str, str | Path],
    schema: DatasetSchema,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_revision: str,
    label_mapping_evidence: str,
    output_dir: str | Path,
    code_revision: str | None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_is_fast: bool | None = None,
    token_counter: Callable[[str], int] | None = None,
    enforce_official_counts: bool = True,
) -> dict:
    """Parse once, audit once, and export stable sidecars.

    The caller must supply a mapping already verified from the official source.
    This function never guesses labels from frequency or column names.
    """
    missing = set(SPLITS) - set(csv_paths)
    if missing:
        raise ValueError(f"Missing CSV paths for {sorted(missing)}")
    splits = {
        split: read_vihsd_csv(csv_paths[split], split=split, schema=schema)
        for split in SPLITS
    }
    if enforce_official_counts:
        mismatches = {
            split: {"expected": EXPECTED_COUNTS[split], "found": len(splits[split])}
            for split in SPLITS if len(splits[split]) != EXPECTED_COUNTS[split]
        }
        if mismatches:
            raise ValueError(f"Official split counts disagree: {mismatches}")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    jsonl_paths = {}
    for split in SPLITS:
        jsonl_paths[split] = save_records_jsonl(
            target / f"{split}.jsonl", splits[split]
        )
    manifest = build_data_manifest(
        parsed_splits=splits,
        csv_paths=csv_paths,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        schema=schema,
        label_mapping_evidence=label_mapping_evidence,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_is_fast=tokenizer_is_fast,
    )
    manifest["prepared_jsonl_sha256"] = {
        split: sha256_file(path) for split, path in jsonl_paths.items()
    }
    manifest_path = write_json(target / "data_manifest.json", manifest)

    report, private = audit_splits(
        splits, phase="integrity_audit", token_counter=token_counter
    )
    audit_paths = save_audit(target / "audit", report=report, private_details=private)
    access_log = target / "test_access.jsonl"
    append_test_access_log(
        access_log,
        phase="integrity_audit",
        code_revision=code_revision,
        purpose="Schema/count/domain/duplicate integrity audit before protocol lock",
        input_hashes={
            split: sha256_file(csv_paths[split]) for split in SPLITS
        },
        outputs=[str(audit_paths["report"]), str(audit_paths["private"])],
    )
    return {
        "output_dir": str(target),
        "manifest_path": str(manifest_path),
        "data_hash": sha256_file(manifest_path),
        "jsonl_paths": {key: str(value) for key, value in jsonl_paths.items()},
        "audit_report_path": str(audit_paths["report"]),
        "test_access_log": str(access_log),
    }


def main() -> None:  # pragma: no cover - integration entry point
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument(
        "--source-label-map",
        required=True,
        help='Verified JSON object, e.g. {"0":0,"1":1,"2":2}',
    )
    parser.add_argument("--label-mapping-evidence", required=True)
    parser.add_argument("--model-id", default="uitnlp/visobert")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-id", default="uitnlp/vihsd")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--code-revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-nonofficial-counts",
        action="store_true",
        help="Fixtures only; never use for official protocol runs",
    )
    args = parser.parse_args()
    mapping = json.loads(args.source_label_map)
    if not isinstance(mapping, dict):
        raise ValueError("--source-label-map must decode to an object")
    result = prepare_csv_bundle(
        csv_paths={
            "train": args.train_csv,
            "validation": args.validation_csv,
            "test": args.test_csv,
        },
        schema=DatasetSchema(
            text_column=args.text_column,
            label_column=args.label_column,
            source_label_to_id={str(key): int(value) for key, value in mapping.items()},
        ),
        model_id=args.model_id,
        model_revision=args.model_revision,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        label_mapping_evidence=args.label_mapping_evidence,
        output_dir=args.output_dir,
        code_revision=args.code_revision,
        enforce_official_counts=not args.allow_nonofficial_counts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
