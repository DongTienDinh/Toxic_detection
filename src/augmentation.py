"""Deterministic training-only AEDA adaptation; see docs/AUGMENTATION_MECHANISM.md.

Inspired by Karimi et al. (2021), https://aclanthology.org/2021.findings-emnlp.234/.
This implementation follows the group's protocol, not the authors' exact recipe.
The core uses only Python's standard library; inject the real locked tokenizer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import re
from typing import Callable, Iterable, Mapping
import unicodedata


PUNCT = (".", ",", "!", "?", ";", ":")
LABELS = (0, 1, 2)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def normalized_key(text: str) -> str:
    """Duplicate comparison only; never replace the model's input with this."""
    return " ".join(unicodedata.normalize("NFC", text).split())


@dataclass(frozen=True)
class AugmentationConfig:
    augmentation_version: str = "aeda_vi_v1"
    augmentation_seed: int = 20261022
    candidate_rate: float = 0.25
    max_length: int = 128

    def __post_init__(self):
        if self.augmentation_version != "aeda_vi_v1":
            raise ValueError("Unsupported augmentation version")
        if type(self.augmentation_seed) is not int:
            raise ValueError("augmentation_seed must be an integer")
        if type(self.candidate_rate) not in (int, float):
            raise ValueError("candidate_rate must be numeric")
        if not 0 <= self.candidate_rate <= 1:
            raise ValueError("candidate_rate must be finite and in [0, 1]")
        if type(self.max_length) is not int or self.max_length < 2:
            raise ValueError("max_length must be an integer >= 2")


def stable_key(purpose: str, source_id: str, cfg: AugmentationConfig) -> str:
    return sha256(f"{purpose}|{cfg.augmentation_version}|"
                  f"{cfg.augmentation_seed}|{source_id}")


def generate_variant(text: str, source_id: str, cfg: AugmentationConfig):
    """Return (text or None, insertion metadata), before length/dedup filters."""
    units = list(re.finditer(r"\S+", text))
    if len(units) < 2:
        return None, []
    row_seed = int(stable_key("generate", source_id, cfg)[:16], 16)
    rng = random.Random(row_seed)
    upper = min(len(units) - 1, max(1, len(units) // 10))
    count = rng.randint(1, upper)
    gaps = sorted(rng.sample(range(len(units) - 1), count), reverse=True)
    out, operations = text, []
    for gap in gaps:
        start, end = units[gap].end(), units[gap + 1].start()
        punct = rng.choice(PUNCT)
        out = out[:start] + " " + punct + " " + out[end:]
        operations.append({"gap_index": gap, "char_start": start,
                           "char_end": end, "punctuation": punct})
    return out, operations


def hf_token_counter(tokenizer) -> Callable[[str], int]:
    """Count real subword tokens, including special tokens, without truncation."""
    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True, truncation=False,
                             padding=False, return_attention_mask=False,
                             return_token_type_ids=False)["input_ids"])
    return count


def _validate_rows(rows: Iterable[Mapping]) -> list[dict]:
    clean, ids = [], set()
    required = ("sample_id", "text", "labels", "split", "source_row")
    for row in rows:
        if any(k not in row for k in required):
            raise ValueError(f"Required fields: {required}")
        r = {k: row[k] for k in required}
        if r["split"] != "train":
            raise ValueError("Augmentation accepts train rows only")
        if type(r["source_row"]) is not int or r["source_row"] < 0:
            raise ValueError("source_row must be a nonnegative 0-based integer")
        if r["sample_id"] != f"train_{r['source_row']:08d}":
            raise ValueError("sample_id does not match the original source_row")
        if r["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        if type(r["labels"]) is not int or r["labels"] not in LABELS:
            raise ValueError("Verify mapping and convert labels to integer 0/1/2")
        if not isinstance(r["text"], str) or not r["text"].strip():
            raise ValueError("Missing, empty or non-string text; audit data first")
        ids.add(r["sample_id"])
        clean.append(r)
    if not clean:
        raise ValueError("Empty train dataset")
    return sorted(clean, key=lambda r: r["sample_id"])


def build_augmented_train(
    rows: Iterable[Mapping],
    token_counter: Callable[[str], int],
    cfg: AugmentationConfig,
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> dict:
    """Build once, reuse across models/seeds. No access to dev/test is needed.

    Input labels must already be verified against the official dataset schema.
    This validates syntax, not truth of the caller's split/mapping declarations.
    """
    if not tokenizer_id or not re.fullmatch(r"[0-9a-fA-F]{40}", tokenizer_revision):
        raise ValueError("Provide tokenizer ID and locked full 40-hex commit SHA")
    original = _validate_rows(rows)
    by_class = {c: [r for r in original if r["labels"] == c] for c in LABELS}
    selected, class_stats = [], {}
    for c, group in by_class.items():
        n = int((Decimal(str(cfg.candidate_rate)) * len(group))
                .to_integral_value(rounding=ROUND_FLOOR))
        ranked = sorted(group, key=lambda r: (stable_key("select", r["sample_id"], cfg),
                                             r["sample_id"]))
        selected.extend(ranked[:n])
        class_stats[str(c)] = {"original": len(group), "candidates": n,
                               "accepted": 0, "rejected": 0, "reasons": {}}

    seen = {normalized_key(r["text"]) for r in original}
    augmented, provenance = [], []

    def length(text):
        n = token_counter(text)
        if type(n) is not int or n <= 0:
            raise ValueError("token_counter must return a positive integer")
        return n

    for row in sorted(selected, key=lambda r: r["sample_id"]):
        sid = row["sample_id"]
        aug_id = f"aug__{cfg.augmentation_version}__{sid}"
        reason, out, ops = None, None, []
        original_length = augmented_length = None
        if len(re.findall(r"\S+", row["text"])) < 2:
            reason = "too_few_units"
        else:
            original_length = length(row["text"])
            if original_length > cfg.max_length:
                reason = "original_too_long"
            else:
                out, ops = generate_variant(row["text"], sid, cfg)
                augmented_length = length(out)
                if augmented_length > cfg.max_length:
                    reason = "augmented_too_long"
                elif normalized_key(out) in seen:
                    reason = "duplicate"
        accepted = reason is None
        stats = class_stats[str(row["labels"])]
        if accepted:
            seen.add(normalized_key(out))
            augmented.append({"sample_id": aug_id, "text": out,
                              "labels": row["labels"], "split": "train",
                              "source_row": row["source_row"], "source_id": sid,
                              "is_augmented": True})
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        provenance.append({
            "aug_id": aug_id, "source_id": sid, "source_split": "train",
            "labels": row["labels"], "operation": cfg.augmentation_version,
            "augmentation_seed": cfg.augmentation_seed,
            "parameters": asdict(cfg), "insertions": ops,
            "original_text_hash": sha256(row["text"]),
            "augmented_text_hash": sha256(out) if out is not None else None,
            "original_length": original_length, "augmented_length": augmented_length,
            "accepted": accepted, "rejection_reason": reason,
        })

    original_output = [{**r, "source_id": r["sample_id"], "is_augmented": False}
                       for r in original]
    combined = original_output + augmented
    audit_pairs, audit_counts = [], {}
    source_lookup = {r["sample_id"]: r for r in original}
    for c in LABELS:
        pool = [r for r in augmented if r["labels"] == c]
        pool.sort(key=lambda r: (stable_key("audit", r["source_id"], cfg),
                                 r["source_id"]))
        chosen = pool[:30]
        audit_counts[str(c)] = len(chosen)
        for r in chosen:
            audit_pairs.append({"aug_id": r["sample_id"], "source_id": r["source_id"],
                                "labels": c, "original_text": source_lookup[r["source_id"]]["text"],
                                "augmented_text": r["text"], "decision": "", "reason": ""})
    manifest = {
        "config": asdict(cfg), "punctuation": list(PUNCT), "punctuation_ratio": 0.1,
        "tokenizer_id": tokenizer_id, "tokenizer_revision": tokenizer_revision,
        "python_version": platform.python_version(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_hash": sha256(canonical_json(original)),
        "augmentation_hash": sha256(canonical_json(augmented)),
        "combined_train_hash": sha256(canonical_json(combined)),
        "n_original": len(original), "n_candidates": len(selected),
        "n_accepted": len(augmented), "n_combined": len(combined),
        "rejections": dict(Counter(p["rejection_reason"] for p in provenance
                                   if not p["accepted"])),
        "per_class": class_stats, "qc_status": "pending_human_review",
        "audit_pairs_per_class": audit_counts,
        "audit_sample_complete": all(n == 30 for n in audit_counts.values()),
    }
    return {"train_original": original_output, "train_augmented": augmented,
            "train_combined": combined, "provenance": provenance,
            "audit_pairs": audit_pairs, "manifest": manifest}


def export_bundle(bundle: dict, output_dir: str | Path) -> Path:
    """Write a new private artifact folder; refuse to overwrite previous output."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    for name in ("train_original", "train_augmented", "train_combined", "provenance"):
        with (target / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in bundle[name]:
                f.write(canonical_json(row) + "\n")
    fields = ["aug_id", "source_id", "labels", "original_text", "augmented_text",
              "decision", "reason"]
    with (target / "audit_pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bundle["audit_pairs"])
    manifest = dict(bundle["manifest"])
    manifest["output_file_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(target.iterdir()) if path.is_file()
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return target
