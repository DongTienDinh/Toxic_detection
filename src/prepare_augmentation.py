"""Build one immutable augmentation bundle for reuse by all heads/seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

from .augmentation import (
    AugmentationConfig,
    build_augmented_train,
    export_bundle as export_aeda_bundle,
    hf_token_counter,
)
from .data import load_records_jsonl
from .io_utils import canonical_json, sha256_file, write_json
from .vietnamese_augmentation import (
    VietnameseAugmentationConfig,
    build_vietnamese_augmented_train,
    load_resource,
)


def export_vietnamese_bundle(bundle: Mapping, output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    for name in ("train_original", "train_augmented", "train_combined", "provenance"):
        with (target / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in bundle[name]:
                handle.write(canonical_json(row) + "\n")
    fields = [
        "aug_id", "source_id", "labels", "selected_operation", "supplemental",
        "original_text", "augmented_text", "decision", "reason",
    ]
    with (target / "audit_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bundle["audit_pairs"])
    manifest = dict(bundle["manifest"])
    manifest["output_file_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(target.iterdir()) if path.is_file()
    }
    write_json(target / "manifest.json", manifest)
    return target


def build_bundle(
    *,
    rows,
    tokenizer,
    tokenizer_id: str,
    tokenizer_revision: str,
    family: str,
    config: Mapping,
    output_dir: str | Path,
    require_locked_resources: bool = True,
) -> Path:
    counter = hf_token_counter(tokenizer)
    if family == "aeda":
        allowed = {"augmentation_version", "augmentation_seed", "candidate_rate", "max_length"}
        cfg = AugmentationConfig(**{key: config[key] for key in allowed})
        bundle = build_augmented_train(
            rows, counter, cfg,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
        )
        return export_aeda_bundle(bundle, output_dir)
    if family != "vietnamese":
        raise ValueError("family must be aeda or vietnamese")
    cfg_fields = {
        "augmentation_version", "augmentation_seed", "candidate_rate",
        "candidate_pool_version", "max_length", "max_edits",
        "teencode_probability_when_both_eligible",
    }
    cfg = VietnameseAugmentationConfig(**{key: config[key] for key in cfg_fields})
    root = Path(config.get("project_root", "."))
    lexicon = load_resource(
        root / config["lexicon_path"], expected_type="teencode_lexicon",
        require_locked=require_locked_resources,
    )
    allowlist = load_resource(
        root / config["allowlist_path"], expected_type="diacritics_allowlist",
        require_locked=require_locked_resources,
    )
    protected = load_resource(
        root / config["protected_terms_path"], expected_type="protected_terms",
        require_locked=require_locked_resources,
    )
    bundle = build_vietnamese_augmented_train(
        rows, counter, cfg,
        lexicon=lexicon,
        allowlist=allowlist,
        protected_terms=protected,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        require_locked_resources=require_locked_resources,
    )
    return export_vietnamese_bundle(bundle, output_dir)


def main() -> None:  # pragma: no cover - exercised on Kaggle
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--family", choices=("aeda", "vietnamese"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-id", default="uitnlp/visobert")
    parser.add_argument("--tokenizer-revision", required=True)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers") from exc
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id, revision=args.tokenizer_revision
    )
    output = build_bundle(
        rows=load_records_jsonl(args.train_jsonl, expected_split="train"),
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        family=args.family,
        config=config,
        output_dir=args.output_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
