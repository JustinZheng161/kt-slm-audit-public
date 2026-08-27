"""Audit train-fitted skill-label support without exporting learner-level records.

The script uses the fixed student-disjoint split and emits only aggregate counts.
It does not save raw labels, user IDs, sequences, or split memberships.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if (HERE.parents[2] / "research_code").exists() else HERE.parents[1]
EXPERIMENTS = ROOT / "research_code" / "experiments" if (ROOT / "research_code").exists() else ROOT / "experiments"
PUBLIC_OUT = (ROOT / "research_code" / "metadata" if (ROOT / "research_code").exists() else ROOT / "metadata") / "skill-label-support-audit.json"
CONTROLLED = Path(os.environ.get("KT_AUDIT_DATA_ROOT", ROOT / "private_data"))
PRIVATE_OUT = CONTROLLED / "metadata" / "skill_label_support_audit_private.json"
sys.path.insert(0, str(EXPERIMENTS))

from run_student_disjoint_kt import SplitSpec, load_student_raw_sequences, split_students  # noqa: E402


def event_and_label_support(sequences: list[list[tuple[str, int]]]) -> tuple[set[str], int]:
    labels = {skill for sequence in sequences for skill, _ in sequence}
    return labels, sum(len(sequence) for sequence in sequences)


def candidate_composites(label_strings: list[str]) -> set[str]:
    """Detect only explicit multi-value separators; underscores are not treated as proof."""
    separators = ("~~", "|||", ";", "|", ",")
    return {label for label in label_strings if any(separator in label for separator in separators)}


def split_composite_summary(sequences: list[list[tuple[str, int]]], candidates: set[str]) -> dict[str, int]:
    unique = {skill for sequence in sequences for skill, _ in sequence if skill in candidates}
    events = sum(1 for sequence in sequences for skill, _ in sequence if skill in candidates)
    return {"candidate_composite_unique_labels": len(unique), "candidate_composite_event_rows": events}


def main() -> None:
    raw = CONTROLLED / "raw" / "skill_builder_data_corrected_collapsed.csv"
    if not raw.exists():
        raise FileNotFoundError(f"Controlled source file not found: {raw}")
    sequences = load_student_raw_sequences(raw)
    label_strings = sorted({skill for sequence in sequences for skill, _ in sequence})
    train, validation, test = split_students(sequences, SplitSpec())
    train_labels, train_events = event_and_label_support(train)
    validation_labels, validation_events = event_and_label_support(validation)
    test_labels, test_events = event_and_label_support(test)
    validation_oov = validation_labels.difference(train_labels)
    test_oov = test_labels.difference(train_labels)
    candidate_ids = candidate_composites(label_strings)
    result = {
        "audit_name": "student_disjoint_skill_label_support",
        "data_file": raw.name,
        "split": {"seed": 20260822, "fractions": {"train": 0.80, "validation": 0.10, "test": 0.10}},
        "preprocessing": {
            "provider_file": "official corrected collapsed ASSISTments2009 CSV",
            "skill_label_rule": "Each unique provider-supplied non-empty skill_id string is treated as an atomic categorical label; no extra merging, renaming, semantic normalization, or delimiter-based decomposition is applied.",
            "candidate_composite_detection": "For disclosure only, labels containing explicit multi-value separators (~~, |||, ;, |, or comma) are counted. An underscore is not considered proof of a composite label.",
            "sequence_filter": "Only student histories with at least two interactions are retained before the student-level split.",
        },
        "student_counts": {"all_retained": len(sequences), "train": len(train), "validation": len(validation), "test": len(test)},
        "event_counts": {"train": train_events, "validation": validation_events, "test": test_events},
        "unique_skill_label_counts": {"all_retained": len(label_strings), "train": len(train_labels), "validation": len(validation_labels), "test": len(test_labels)},
        "train_fitted_vocabulary_support": {
            "validation_labels_not_seen_in_train": len(validation_oov),
            "test_labels_not_seen_in_train": len(test_oov),
            "validation_label_coverage_by_train": len(validation_labels.intersection(train_labels)) / len(validation_labels) if validation_labels else None,
            "test_label_coverage_by_train": len(test_labels.intersection(train_labels)) / len(test_labels) if test_labels else None,
            "validation_event_rows_with_unseen_label": sum(1 for sequence in validation for skill, _ in sequence if skill in validation_oov),
            "test_event_rows_with_unseen_label": sum(1 for sequence in test for skill, _ in sequence if skill in test_oov),
        },
        "candidate_composite_label_distribution": {
            "overall_candidate_composite_unique_labels": len(candidate_ids),
            "train": split_composite_summary(train, candidate_ids),
            "validation": split_composite_summary(validation, candidate_ids),
            "test": split_composite_summary(test, candidate_ids),
        },
        "assessment": "Pass only if both validation_labels_not_seen_in_train and test_labels_not_seen_in_train equal zero. Otherwise a coverage-constrained split or explicit OOV policy is required before main-result reporting.",
        "privacy": "Only aggregate counts are written. No raw skill strings, student IDs, split memberships, sequences, or row-level predictions are exported.",
    }
    for out in (PRIVATE_OUT, PUBLIC_OUT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
