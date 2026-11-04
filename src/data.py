"""ViHSD download, strict CSV parsing and tokenization utilities.

This module never infers a label mapping from frequency. Callers must provide a
mapping that has already been verified against the official dataset evidence.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

from .io_utils import canonical_json, read_jsonl, sha256_file, sha256_text, write_jsonl


SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 24048, "validation": 2672, "test": 6680}
FULL_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class DatasetSchema:
    text_column: str
    label_column: str
    source_label_to_id: Mapping[str, int]

    def validate(self) -> None:
        if not self.text_column or not self.label_column:
            raise ValueError("text_column and label_column are required")
        if self.text_column == self.label_column:
            raise ValueError("text and label columns must differ")
        values = sorted(self.source_label_to_id.values())
        if values != [0, 1, 2] or len(self.source_label_to_id) != 3:
            raise ValueError("Verified source mapping must cover exactly IDs 0,1,2")
        if any(not isinstance(key, str) for key in self.source_label_to_id):
            raise ValueError("CSV source-label keys must be strings")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    text: str
    labels: int
    split: str
    source_row: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def read_vihsd_csv(
    path: str | Path,
    *,
    split: str,
    schema: DatasetSchema,
    encoding: str = "utf-8-sig",
) -> list[dict]:
    """Parse a CSV without changing row order or raw text field content."""
    _validate_split(split)
    schema.validate()
    records: list[dict] = []
    with Path(path).open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = {schema.text_column, schema.label_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {path}")
        for source_row, row in enumerate(reader):
            text = row.get(schema.text_column)
            raw_label = row.get(schema.label_column)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Null/empty text at {split} source_row={source_row}")
            if raw_label is None:
                raise ValueError(f"Null label at {split} source_row={source_row}")
            label_key = str(raw_label).strip()
            if label_key not in schema.source_label_to_id:
                raise ValueError(
                    f"Unverified label {label_key!r} at {split} source_row={source_row}"
                )
            record = SampleRecord(
                sample_id=f"{split}_{source_row:08d}",
                text=text,
                labels=int(schema.source_label_to_id[label_key]),
                split=split,
                source_row=source_row,
            )
            records.append(record.to_dict())
    if not records:
        raise ValueError(f"Empty CSV: {path}")
    validate_records(records, expected_split=split)
    return records


def validate_records(records: Sequence[Mapping], *, expected_split: str | None = None) -> None:
    if not records:
        raise ValueError("Dataset records are empty")
    seen: set[str] = set()
    required = {"sample_id", "text", "labels", "split", "source_row"}
    for position, record in enumerate(records):
        if not required.issubset(record):
            raise ValueError(f"Record {position} lacks {sorted(required - set(record))}")
        split = record["split"]
        _validate_split(split)
        if expected_split is not None and split != expected_split:
            raise ValueError(f"Expected split={expected_split}, found {split}")
        source_row = record["source_row"]
        if type(source_row) is not int or source_row < 0:
            raise ValueError("source_row must be a nonnegative integer")
        expected_id = f"{split}_{source_row:08d}"
        if record["sample_id"] != expected_id:
            raise ValueError(f"sample_id must be {expected_id}")
        if record["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
        seen.add(record["sample_id"])
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"Invalid text at {record['sample_id']}")
        if type(record["labels"]) is not int or record["labels"] not in (0, 1, 2):
            raise ValueError(f"Invalid label at {record['sample_id']}")


def load_records_jsonl(path: str | Path, *, expected_split: str | None = None) -> list[dict]:
    records = read_jsonl(path)
    validate_records(records, expected_split=expected_split)
    return records


def validate_training_records(records: Sequence[Mapping]) -> None:
    """Validate original or accepted augmented train rows.

    Augmented IDs intentionally do not use the original split/source-row ID
    format, but must retain a valid ``source_id`` pointer.
    """
    if not records:
        raise ValueError("Training records are empty")
    seen = set()
    for position, record in enumerate(records):
        required = {"sample_id", "text", "labels", "split", "source_row"}
        if not required.issubset(record):
            raise ValueError(f"Training row {position} lacks required fields")
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Invalid/duplicate training sample_id: {sample_id!r}")
        seen.add(sample_id)
        if record["split"] != "train":
            raise ValueError("Training corpus may contain train rows only")
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"Invalid text at {sample_id}")
        if type(record["labels"]) is not int or record["labels"] not in (0, 1, 2):
            raise ValueError(f"Invalid label at {sample_id}")
        if type(record["source_row"]) is not int or record["source_row"] < 0:
            raise ValueError(f"Invalid source_row at {sample_id}")
        augmented = bool(record.get("is_augmented", False))
        if augmented:
            source_id = record.get("source_id")
            if source_id != f"train_{record['source_row']:08d}":
                raise ValueError(f"Augmented row {sample_id} has invalid source_id")
            if not sample_id.startswith("aug__"):
                raise ValueError(f"Augmented row {sample_id} needs aug__ prefix")
        elif sample_id != f"train_{record['source_row']:08d}":
            raise ValueError(f"Original row ID mismatch: {sample_id}")


def load_training_jsonl(path: str | Path) -> list[dict]:
    records = read_jsonl(path)
    validate_training_records(records)
    return records


def save_records_jsonl(path: str | Path, records: Sequence[Mapping]) -> Path:
    validate_records(records)
    return write_jsonl(path, records)


def records_hash(records: Sequence[Mapping]) -> str:
    validate_records(records)
    return sha256_text(canonical_json([dict(record) for record in records]))


def training_records_hash(records: Sequence[Mapping]) -> str:
    validate_training_records(records)
    return sha256_text(canonical_json([dict(record) for record in records]))


def download_hf_assets(
    *,
    model_id: str,
    dataset_id: str,
    token: str,
    model_revision: str | None = None,
    dataset_revision: str | None = None,
) -> dict:
    """Resolve revisions once and download official ViHSD CSV files.

    Tokens are accepted only as an argument and are never returned or logged.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - bootstrap dependent
        raise RuntimeError("Install huggingface_hub") from exc
    if not token:
        raise ValueError("A read token is required for the gated dataset")
    api = HfApi(token=token)
    model_sha = model_revision or api.model_info(model_id).sha
    dataset_sha = dataset_revision or api.dataset_info(dataset_id).sha
    if not FULL_SHA_PATTERN.fullmatch(str(model_sha)):
        raise ValueError("Model revision did not resolve to a full SHA")
    if not FULL_SHA_PATTERN.fullmatch(str(dataset_sha)):
        raise ValueError("Dataset revision did not resolve to a full SHA")
    files = {}
    for split, filename in {
        "train": "train.csv",
        "validation": "dev.csv",
        "test": "test.csv",
    }.items():
        files[split] = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            revision=dataset_sha,
            token=token,
        )
    return {
        "model_id": model_id,
        "model_revision": model_sha,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_sha,
        "files": files,
    }


def build_data_manifest(
    *,
    parsed_splits: Mapping[str, Sequence[Mapping]],
    csv_paths: Mapping[str, str | Path],
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_revision: str,
    schema: DatasetSchema,
    label_mapping_evidence: str,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_is_fast: bool | None = None,
) -> dict:
    if not label_mapping_evidence.strip():
        raise ValueError("label_mapping_evidence is required")
    for split in SPLITS:
        if split not in parsed_splits or split not in csv_paths:
            raise ValueError(f"Missing split {split}")
        validate_records(parsed_splits[split], expected_split=split)
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "tokenizer": {
            "id": tokenizer_id or model_id,
            "revision": tokenizer_revision or model_revision,
            "is_fast": tokenizer_is_fast,
        },
        "schema": {
            "text_column": schema.text_column,
            "label_column": schema.label_column,
            "source_label_to_id": dict(schema.source_label_to_id),
            "label_mapping_evidence": label_mapping_evidence,
        },
        "splits": {
            split: {
                "rows": len(parsed_splits[split]),
                "expected_rows": EXPECTED_COUNTS[split],
                "count_matches_expected": len(parsed_splits[split]) == EXPECTED_COUNTS[split],
                "csv_sha256": sha256_file(csv_paths[split]),
                "records_sha256": records_hash(parsed_splits[split]),
            }
            for split in SPLITS
        },
    }


def to_hf_dataset(
    records: Sequence[Mapping], tokenizer, *, max_length: int = 128,
    allow_augmented_train: bool = False,
):
    """Create a tokenized Dataset containing only model inputs and labels."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - bootstrap dependent
        raise RuntimeError("Install datasets") from exc
    if allow_augmented_train:
        validate_training_records(records)
    else:
        validate_records(records)
    dataset = Dataset.from_list([
        {"text": record["text"], "labels": record["labels"]} for record in records
    ])

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length,
            padding=False, return_token_type_ids=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    allowed = {"input_ids", "attention_mask", "labels"}
    remove = [column for column in tokenized.column_names if column not in allowed]
    return tokenized.remove_columns(remove) if remove else tokenized
