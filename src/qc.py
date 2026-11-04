"""Human label-preservation QC for immutable augmentation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Mapping

from .io_utils import read_json, sha256_file, write_json


DECISIONS = {"preserved", "changed", "uncertain"}


def _read_review(path: str | Path) -> dict[str, dict]:
    rows = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"aug_id", "labels", "decision", "reason"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Review CSV lacks columns {sorted(required)}: {path}")
        for line, row in enumerate(reader, start=2):
            aug_id = row["aug_id"].strip()
            decision = row["decision"].strip().lower()
            if not aug_id or aug_id in rows:
                raise ValueError(f"Missing/duplicate aug_id at {path}:{line}")
            if decision not in DECISIONS:
                raise ValueError(f"Invalid decision at {path}:{line}: {decision!r}")
            try:
                label = int(row["labels"])
            except ValueError as exc:
                raise ValueError(f"Invalid label at {path}:{line}") from exc
            if label not in (0, 1, 2):
                raise ValueError(f"Out-of-domain label at {path}:{line}")
            rows[aug_id] = {
                "labels": label,
                "decision": decision,
                "reason": row["reason"].strip(),
            }
    if not rows:
        raise ValueError(f"Empty review: {path}")
    return rows


def _read_adjudication(path: str | Path | None) -> dict[str, dict]:
    return {} if path is None else _read_review(path)


def build_qc_report(
    *,
    manifest_path: str | Path,
    reviewer_1_path: str | Path,
    reviewer_2_path: str | Path,
    reviewer_1_id: str,
    reviewer_2_id: str,
    adjudication_path: str | Path | None,
    adjudicator_id: str | None,
    systematic_error_found: bool,
    minimum_preservation_rate: float = 0.95,
    required_per_class: int = 30,
) -> dict:
    if not reviewer_1_id.strip() or not reviewer_2_id.strip():
        raise ValueError("Both reviewer identifiers are required")
    if reviewer_1_id.strip() == reviewer_2_id.strip():
        raise ValueError("Reviewers must be two distinct people")
    if not 0 < minimum_preservation_rate <= 1:
        raise ValueError("minimum_preservation_rate must be in (0,1]")
    if type(required_per_class) is not int or required_per_class <= 0:
        raise ValueError("required_per_class must be positive")

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Augmentation manifest must be an object")
    if manifest.get("qc_status") != "pending_human_review":
        raise ValueError("QC must start from pending_human_review manifest")
    first = _read_review(reviewer_1_path)
    second = _read_review(reviewer_2_path)
    if set(first) != set(second):
        raise ValueError("Reviewer files must contain identical aug_id sets")
    for aug_id in first:
        if first[aug_id]["labels"] != second[aug_id]["labels"]:
            raise ValueError(f"Reviewer label metadata differs for {aug_id}")

    per_class = Counter(row["labels"] for row in first.values())
    enough = all(per_class[label] == required_per_class for label in (0, 1, 2))
    agreements = {
        aug_id for aug_id in first
        if first[aug_id]["decision"] == second[aug_id]["decision"]
    }
    disagreements = sorted(set(first) - agreements)
    adjudication = _read_adjudication(adjudication_path)
    if disagreements:
        if not adjudicator_id or not adjudicator_id.strip():
            raise ValueError("An adjudicator is required for reviewer disagreements")
        if set(adjudication) != set(disagreements):
            raise ValueError("Adjudication must contain exactly the disagreement IDs")
        for aug_id in disagreements:
            if adjudication[aug_id]["labels"] != first[aug_id]["labels"]:
                raise ValueError(f"Adjudication label metadata differs for {aug_id}")
    elif adjudication:
        raise ValueError("Adjudication supplied although reviewers fully agree")

    final_decisions = {
        aug_id: (
            first[aug_id]["decision"]
            if aug_id in agreements else adjudication[aug_id]["decision"]
        )
        for aug_id in first
    }
    total = len(final_decisions)
    preserved = sum(value == "preserved" for value in final_decisions.values())
    preservation_rate = preserved / total
    threshold_count = math.ceil(minimum_preservation_rate * total)
    if not enough:
        status = "insufficient_samples"
    elif systematic_error_found or preserved < threshold_count:
        status = "failed"
    else:
        status = "passed"
    final_counts = Counter(final_decisions.values())
    return {
        "schema_version": 1,
        "qc_status": status,
        "source_manifest_sha256": sha256_file(manifest_path),
        "augmentation_hash": manifest.get("augmentation_hash"),
        "combined_train_hash": manifest.get("combined_train_hash"),
        "reviewers": [reviewer_1_id.strip(), reviewer_2_id.strip()],
        "adjudicator": adjudicator_id.strip() if adjudicator_id else None,
        "review_file_sha256": {
            "reviewer_1": sha256_file(reviewer_1_path),
            "reviewer_2": sha256_file(reviewer_2_path),
            "adjudication": sha256_file(adjudication_path) if adjudication_path else None,
        },
        "required_per_class": required_per_class,
        "sample_count_per_class": {str(key): per_class[key] for key in (0, 1, 2)},
        "sample_count": total,
        "agreement_count": len(agreements),
        "raw_agreement": len(agreements) / total,
        "disagreement_count": len(disagreements),
        "final_decision_counts": dict(sorted(final_counts.items())),
        "minimum_preservation_rate": minimum_preservation_rate,
        "minimum_preserved_count": threshold_count,
        "preservation_rate": preservation_rate,
        "systematic_error_found": bool(systematic_error_found),
    }


def approve_manifest(
    manifest_path: str | Path,
    qc_report_path: str | Path,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    report_path = Path(qc_report_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if report_path.resolve().parent != output.resolve().parent:
        raise ValueError("QC report and approved manifest must share an artifact directory")
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise ValueError("Manifest and QC report must be objects")
    if report.get("qc_status") != "passed":
        raise ValueError("Only a passed QC report can approve augmentation")
    if report.get("source_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("QC report was produced from a different manifest")
    for key in ("augmentation_hash", "combined_train_hash"):
        if report.get(key) != manifest.get(key):
            raise ValueError(f"QC report {key} mismatch")
    approved = dict(manifest)
    approved["qc_status"] = "passed"
    approved["qc_report_sha256"] = sha256_file(report_path)
    approved["qc_report_path"] = report_path.name
    return write_json(output, approved)


def main() -> None:  # pragma: no cover - human workflow entry point
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--manifest", required=True)
    review.add_argument("--reviewer-1", required=True)
    review.add_argument("--reviewer-2", required=True)
    review.add_argument("--reviewer-1-id", required=True)
    review.add_argument("--reviewer-2-id", required=True)
    review.add_argument("--adjudication")
    review.add_argument("--adjudicator-id")
    review.add_argument("--systematic-error-found", action="store_true")
    review.add_argument("--output", required=True)
    approve = subparsers.add_parser("approve")
    approve.add_argument("--manifest", required=True)
    approve.add_argument("--qc-report", required=True)
    approve.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "review":
        if Path(args.output).exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        report = build_qc_report(
            manifest_path=args.manifest,
            reviewer_1_path=args.reviewer_1,
            reviewer_2_path=args.reviewer_2,
            reviewer_1_id=args.reviewer_1_id,
            reviewer_2_id=args.reviewer_2_id,
            adjudication_path=args.adjudication,
            adjudicator_id=args.adjudicator_id,
            systematic_error_found=args.systematic_error_found,
        )
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(approve_manifest(args.manifest, args.qc_report, args.output))


if __name__ == "__main__":
    main()
