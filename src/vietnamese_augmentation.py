"""Deterministic Vietnamese social-text augmentation for protocol v2.

The module implements the protocol contract, but it deliberately refuses
unreviewed resources for production corpus generation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import platform
import random
import re
from typing import Callable, Iterable, Mapping, Sequence
import unicodedata

from .data import validate_records
from .io_utils import canonical_json, sha256_file, sha256_text


SUPPORTED_RECIPES = (
    "vi_teencode_v1",
    "vi_partial_diacritics_v1",
    "vi_social_aug_v1",
)
LABELS = (0, 1, 2)
URL_EMAIL_HANDLE = re.compile(
    r"https?://\S+|www\.\S+|(?<!\w)[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\w)@[\w_]+",
    re.UNICODE | re.IGNORECASE,
)


@dataclass(frozen=True)
class Unit:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Match:
    operation: str
    entry_id: str
    start: int
    end: int
    before: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class VietnameseAugmentationConfig:
    augmentation_version: str = "vi_social_aug_v1"
    augmentation_seed: int = 20261022
    candidate_rate: float = 0.25
    candidate_pool_version: str = "vi_candidate_pool_v1"
    max_length: int = 128
    max_edits: int = 1
    teencode_probability_when_both_eligible: float = 0.5

    def __post_init__(self):
        if self.augmentation_version not in SUPPORTED_RECIPES:
            raise ValueError(f"Unsupported recipe: {self.augmentation_version}")
        if type(self.augmentation_seed) is not int:
            raise ValueError("augmentation_seed must be int")
        if type(self.candidate_rate) not in (int, float) or not 0 <= self.candidate_rate <= 1:
            raise ValueError("candidate_rate must be finite and in [0,1]")
        if self.candidate_pool_version != "vi_candidate_pool_v1":
            raise ValueError("Unsupported candidate_pool_version")
        if type(self.max_length) is not int or self.max_length < 2:
            raise ValueError("max_length must be an integer >=2")
        if self.max_edits != 1:
            raise ValueError("Protocol v2 supports exactly one edit")
        if self.teencode_probability_when_both_eligible != 0.5:
            raise ValueError("Protocol v2 mixture probability must be 0.5")


def _is_unit_character(character: str) -> bool:
    return character == "_" or unicodedata.category(character)[0] in {"L", "M", "N"}


def scan_units(text: str) -> list[Unit]:
    units: list[Unit] = []
    start = None
    for index, character in enumerate(text):
        if _is_unit_character(character):
            if start is None:
                start = index
        elif start is not None:
            units.append(Unit(text[start:index], start, index))
            start = None
    if start is not None:
        units.append(Unit(text[start:], start, len(text)))
    return units


def remove_vietnamese_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", stripped)


def _resource_entries(resource: Mapping, *, expected_type: str, require_locked: bool) -> list[dict]:
    if resource.get("resource_type") != expected_type:
        raise ValueError(f"Expected resource_type={expected_type}")
    if require_locked and resource.get("status") not in {"approved", "locked"}:
        raise ValueError(f"{expected_type} is not approved/locked")
    entries = resource.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{expected_type}.entries must be a list")
    approved = []
    seen_ids = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("Resource entries must be objects")
        entry_id = raw.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("Every resource entry needs entry_id")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate entry_id: {entry_id}")
        seen_ids.add(entry_id)
        if raw.get("review_status") != "approved":
            continue
        reviewers = raw.get("reviewers")
        if not isinstance(reviewers, list) or not reviewers or not all(reviewers):
            raise ValueError(f"Approved entry {entry_id} lacks reviewer IDs")
        approved.append(dict(raw))
    return approved


def load_resource(path: str | Path, *, expected_type: str, require_locked: bool = True) -> dict:
    target = Path(path)
    resource = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(resource, dict):
        raise ValueError("Resource root must be an object")
    _resource_entries(resource, expected_type=expected_type, require_locked=require_locked)
    resource = dict(resource)
    resource["file_sha256"] = sha256_file(target)
    return resource


def resource_hash(resource: Mapping) -> str:
    payload = {key: value for key, value in resource.items() if key != "file_sha256"}
    return sha256_text(canonical_json(payload))


def _units_for_key(value: str) -> list[Unit]:
    units = scan_units(value)
    if not units:
        raise ValueError("Resource key contains no units")
    for left, right in zip(units, units[1:]):
        if not value[left.end:right.start].isspace():
            raise ValueError("Multi-unit resource keys must be whitespace-separated")
    return units


def _automatic_protected_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in URL_EMAIL_HANDLE.finditer(text)]


def _overlaps(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start < protected_end and end > protected_start
               for protected_start, protected_end in ranges)


def _term_sequences(resource: Mapping, *, require_locked: bool) -> list[tuple[str, list[str]]]:
    output = []
    for entry in _resource_entries(
        resource, expected_type="protected_terms", require_locked=require_locked
    ):
        term = entry.get("term", entry.get("source_text"))
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"Protected entry {entry['entry_id']} lacks term")
        units = _units_for_key(unicodedata.normalize("NFC", term))
        output.append((entry["entry_id"], [u.text.casefold() for u in units]))
    return output


def protected_ranges(
    text: str,
    units: Sequence[Unit],
    protected_resource: Mapping,
    *,
    require_locked: bool,
) -> list[tuple[int, int]]:
    ranges = _automatic_protected_ranges(text)
    for _, sequence in _term_sequences(protected_resource, require_locked=require_locked):
        width = len(sequence)
        for index in range(len(units) - width + 1):
            selected = units[index:index + width]
            if [unicodedata.normalize("NFC", unit.text).casefold() for unit in selected] != sequence:
                continue
            if any(not text[left.end:right.start].isspace()
                   for left, right in zip(selected, selected[1:])):
                continue
            ranges.append((selected[0].start, selected[-1].end))
    return sorted(set(ranges))


def _select_non_overlapping(matches: Sequence[Match]) -> list[Match]:
    ranked = sorted(matches, key=lambda item: (item.start, -(item.end - item.start), item.entry_id))
    selected: list[Match] = []
    cursor = -1
    for match in ranked:
        if match.start < cursor:
            continue
        selected.append(match)
        cursor = match.end
    return selected


def find_teencode_matches(
    text: str,
    units: Sequence[Unit],
    lexicon: Mapping,
    protected: Sequence[tuple[int, int]],
    *,
    require_locked: bool,
) -> list[Match]:
    matches = []
    for entry in _resource_entries(
        lexicon, expected_type="teencode_lexicon", require_locked=require_locked
    ):
        source = entry.get("source_text")
        variants = entry.get("variants")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Lexicon entry {entry['entry_id']} lacks source_text")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Lexicon entry {entry['entry_id']} lacks variants")
        if any(not isinstance(value, str) or not value.strip() for value in variants):
            raise ValueError("Variants must be non-empty strings")
        if len(set(variants)) != len(variants):
            raise ValueError(f"Duplicate variants in {entry['entry_id']}")
        source_units = _units_for_key(unicodedata.normalize("NFC", source))
        if not 1 <= len(source_units) <= 3:
            raise ValueError("Teencode source span must contain 1-3 units")
        expected = [unit.text for unit in source_units]
        for index in range(len(units) - len(expected) + 1):
            selected = units[index:index + len(expected)]
            if [unicodedata.normalize("NFC", unit.text) for unit in selected] != expected:
                continue
            if any(not text[left.end:right.start].isspace()
                   for left, right in zip(selected, selected[1:])):
                continue
            start, end = selected[0].start, selected[-1].end
            if _overlaps(start, end, protected):
                continue
            before = text[start:end]
            valid_variants = tuple(value for value in variants if value != before)
            if valid_variants:
                matches.append(Match(
                    "teencode", entry["entry_id"], start, end, before, valid_variants
                ))
    return _select_non_overlapping(matches)


def find_diacritic_matches(
    text: str,
    units: Sequence[Unit],
    allowlist: Mapping,
    protected: Sequence[tuple[int, int]],
    *,
    require_locked: bool,
) -> list[Match]:
    allowed = {}
    for entry in _resource_entries(
        allowlist, expected_type="diacritics_allowlist", require_locked=require_locked
    ):
        source = entry.get("source_text")
        if not isinstance(source, str) or len(_units_for_key(source)) != 1:
            raise ValueError("Diacritics allowlist entries must contain one unit")
        key = unicodedata.normalize("NFC", source)
        if key in allowed:
            raise ValueError(f"Duplicate allowlist source: {key}")
        allowed[key] = entry["entry_id"]
    matches = []
    for unit in units:
        before = unicodedata.normalize("NFC", unit.text)
        if before not in allowed or _overlaps(unit.start, unit.end, protected):
            continue
        after = remove_vietnamese_diacritics(unit.text)
        if after == unit.text:
            continue
        matches.append(Match(
            "diacritics", allowed[before], unit.start, unit.end, unit.text, (after,)
        ))
    return matches


def _stable_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _selection_key(source_id: str, cfg: VietnameseAugmentationConfig) -> str:
    return hashlib.sha256(
        f"select|{cfg.candidate_pool_version}|{cfg.augmentation_seed}|{source_id}".encode("utf-8")
    ).hexdigest()


def _row_rng(source_id: str, cfg: VietnameseAugmentationConfig) -> random.Random:
    digest = _stable_digest(
        f"generate|{cfg.augmentation_version}|{cfg.augmentation_seed}|{source_id}"
    )
    return random.Random(int.from_bytes(digest[:8], "big"))


def _choose_operation(
    teencode: Sequence[Match],
    diacritics: Sequence[Match],
    cfg: VietnameseAugmentationConfig,
    rng: random.Random,
) -> tuple[str | None, Sequence[Match]]:
    recipe = cfg.augmentation_version
    if recipe == "vi_teencode_v1":
        return ("teencode", teencode) if teencode else (None, ())
    if recipe == "vi_partial_diacritics_v1":
        return ("diacritics", diacritics) if diacritics else (None, ())
    if teencode and diacritics:
        return ("teencode", teencode) if rng.random() < 0.5 else ("diacritics", diacritics)
    if teencode:
        return "teencode", teencode
    if diacritics:
        return "diacritics", diacritics
    return None, ()


def build_vietnamese_augmented_train(
    rows: Iterable[Mapping],
    token_counter: Callable[[str], int],
    cfg: VietnameseAugmentationConfig,
    *,
    lexicon: Mapping,
    allowlist: Mapping,
    protected_terms: Mapping,
    tokenizer_id: str,
    tokenizer_revision: str,
    require_locked_resources: bool = True,
) -> dict:
    original = [dict(row) for row in rows]
    validate_records(original, expected_split="train")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", tokenizer_revision):
        raise ValueError("tokenizer_revision must be a locked full SHA")
    if not tokenizer_id:
        raise ValueError("tokenizer_id is required")
    # Validate all resources before selection to avoid partially built corpora.
    _resource_entries(lexicon, expected_type="teencode_lexicon",
                      require_locked=require_locked_resources)
    _resource_entries(allowlist, expected_type="diacritics_allowlist",
                      require_locked=require_locked_resources)
    _resource_entries(protected_terms, expected_type="protected_terms",
                      require_locked=require_locked_resources)

    original.sort(key=lambda row: row["sample_id"])
    by_class = {label: [row for row in original if row["labels"] == label]
                for label in LABELS}
    selected, per_class = [], {}
    for label, group in by_class.items():
        count = int((Decimal(str(cfg.candidate_rate)) * len(group)).to_integral_value(
            rounding=ROUND_FLOOR
        ))
        ranked = sorted(group, key=lambda row: (
            _selection_key(row["sample_id"], cfg), row["sample_id"]
        ))
        selected.extend(ranked[:count])
        per_class[str(label)] = {
            "original": len(group), "candidates": count,
            "accepted": 0, "rejected": 0, "reasons": {},
            "selected_operations": {}, "accepted_operations": {},
        }

    seen = {" ".join(unicodedata.normalize("NFC", row["text"]).split())
            for row in original}
    augmented, provenance = [], []

    def length(text: str) -> int:
        value = token_counter(text)
        if type(value) is not int or value <= 0:
            raise ValueError("token_counter must return a positive integer")
        return value

    for row in sorted(selected, key=lambda item: item["sample_id"]):
        source_id = row["sample_id"]
        rng = _row_rng(source_id, cfg)
        units = scan_units(row["text"])
        protected = protected_ranges(
            row["text"], units, protected_terms, require_locked=require_locked_resources
        )
        teen = find_teencode_matches(
            row["text"], units, lexicon, protected, require_locked=require_locked_resources
        )
        accent = find_diacritic_matches(
            row["text"], units, allowlist, protected, require_locked=require_locked_resources
        )
        operation, eligible = _choose_operation(teen, accent, cfg, rng)
        reason = None
        chosen = None
        output = None
        original_length = augmented_length = None
        if operation is None:
            reason = "no_eligible_span"
        else:
            chosen = eligible[rng.randrange(len(eligible))]
            after = chosen.variants[rng.randrange(len(chosen.variants))]
            original_length = length(row["text"])
            if original_length > cfg.max_length:
                reason = "original_too_long"
            else:
                output = row["text"][:chosen.start] + after + row["text"][chosen.end:]
                if output == row["text"]:
                    reason = "no_change"
                else:
                    augmented_length = length(output)
                    if augmented_length > cfg.max_length:
                        reason = "augmented_too_long"
                    else:
                        duplicate_key = " ".join(unicodedata.normalize("NFC", output).split())
                        if duplicate_key in seen:
                            reason = "duplicate"
        accepted = reason is None
        stats = per_class[str(row["labels"])]
        if operation is not None:
            stats["selected_operations"][operation] = (
                stats["selected_operations"].get(operation, 0) + 1
            )
        aug_id = f"aug__{cfg.augmentation_version}__{source_id}"
        after_value = None
        if chosen is not None and output is not None:
            after_value = output[chosen.start:chosen.start + (len(output) - len(row["text"]) + len(chosen.before))]
        if accepted:
            duplicate_key = " ".join(unicodedata.normalize("NFC", output).split())
            seen.add(duplicate_key)
            augmented.append({
                "sample_id": aug_id,
                "text": output,
                "labels": row["labels"],
                "split": "train",
                "source_row": row["source_row"],
                "source_id": source_id,
                "is_augmented": True,
                "selected_operation": operation,
            })
            stats["accepted"] += 1
            stats["accepted_operations"][operation] = (
                stats["accepted_operations"].get(operation, 0) + 1
            )
        else:
            stats["rejected"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        provenance.append({
            "aug_id": aug_id,
            "source_id": source_id,
            "source_split": "train",
            "labels": row["labels"],
            "operation": cfg.augmentation_version,
            "selected_operation": operation,
            "entry_id": chosen.entry_id if chosen else None,
            "raw_span_start": chosen.start if chosen else None,
            "raw_span_end": chosen.end if chosen else None,
            "before": chosen.before if chosen else None,
            "after": after_value,
            "candidate_pool_version": cfg.candidate_pool_version,
            "augmentation_seed": cfg.augmentation_seed,
            "parameters": asdict(cfg),
            "original_text_hash": sha256_text(row["text"]),
            "augmented_text_hash": sha256_text(output) if output is not None else None,
            "original_length": original_length,
            "augmented_length": augmented_length,
            "eligible_teencode": len(teen),
            "eligible_diacritics": len(accent),
            "accepted": accepted,
            "rejection_reason": reason,
        })

    original_output = [
        {**row, "source_id": row["sample_id"], "is_augmented": False}
        for row in original
    ]
    combined = original_output + augmented
    source_lookup = {row["sample_id"]: row for row in original}
    audit_pairs, audit_counts, supplemental_counts = [], {}, {}
    for label in LABELS:
        pool = [row for row in augmented if row["labels"] == label]
        pool.sort(key=lambda row: (
            hashlib.sha256(
                f"audit|{cfg.augmentation_version}|{cfg.augmentation_seed}|{row['source_id']}".encode("utf-8")
            ).hexdigest(),
            row["source_id"],
        ))
        chosen_rows = pool[:30]
        audit_counts[str(label)] = len(chosen_rows)
        supplements = []
        if cfg.augmentation_version == "vi_social_aug_v1":
            represented = {row["selected_operation"] for row in chosen_rows}
            available = {row["selected_operation"] for row in pool}
            for operation in sorted(available - represented):
                candidate = next(
                    row for row in pool
                    if row["selected_operation"] == operation and row not in chosen_rows
                )
                supplements.append(candidate)
        supplemental_counts[str(label)] = len(supplements)
        for augmented_row in [*chosen_rows, *supplements]:
            audit_pairs.append({
                "aug_id": augmented_row["sample_id"],
                "source_id": augmented_row["source_id"],
                "labels": label,
                "selected_operation": augmented_row["selected_operation"],
                "supplemental": augmented_row in supplements,
                "original_text": source_lookup[augmented_row["source_id"]]["text"],
                "augmented_text": augmented_row["text"],
                "decision": "",
                "reason": "",
            })

    manifest = {
        "config": asdict(cfg),
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "python_version": platform.python_version(),
        "implementation_sha256": sha256_file(Path(__file__)),
        "resource_hashes": {
            "lexicon": resource_hash(lexicon),
            "allowlist": resource_hash(allowlist),
            "protected_terms": resource_hash(protected_terms),
        },
        "resource_versions": {
            "lexicon": lexicon.get("version"),
            "allowlist": allowlist.get("version"),
            "protected_terms": protected_terms.get("version"),
        },
        "train_hash": sha256_text(canonical_json(original)),
        "augmentation_hash": sha256_text(canonical_json(augmented)),
        "combined_train_hash": sha256_text(canonical_json(combined)),
        "n_original": len(original),
        "n_candidates": len(selected),
        "n_accepted": len(augmented),
        "n_combined": len(combined),
        "rejections": dict(Counter(
            item["rejection_reason"] for item in provenance if not item["accepted"]
        )),
        "per_class": per_class,
        "qc_status": "pending_human_review",
        "audit_pairs_per_class": audit_counts,
        "audit_supplements_per_class": supplemental_counts,
        "audit_sample_complete": all(value == 30 for value in audit_counts.values()),
        "test_only_resources": not require_locked_resources,
    }
    return {
        "train_original": original_output,
        "train_augmented": augmented,
        "train_combined": combined,
        "provenance": provenance,
        "audit_pairs": audit_pairs,
        "manifest": manifest,
    }
