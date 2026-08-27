"""Profile history lengths without writing learner-level records.

The output reports only split-level counts, length quantiles and the proportions of
student histories exceeding pre-specified context thresholds. It is designed for the
Revision 5 window-mismatch response and never writes identifiers, memberships,
sequences or predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]
EXPERIMENTS = ROOT / "research_code" / "experiments" if (ROOT / "research_code").exists() else ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_student_disjoint_kt import RAW_DATA, SplitSpec, load_train_fitted_splits, stable_sha256


def archive_result_directory(root: Path) -> Path:
    if (root / "research_code").exists():
        return root / "research_code" / "results"
    if (root / "results" / "revision3").exists():
        return root / "results" / "revision3"
    return root / "artifacts" / "revision3"


ARCHIVE_RESULTS = archive_result_directory(ROOT)
CONTROLLED_RESULTS = RAW_DATA.parents[1] / "results"
QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.99)
THRESHOLDS = (200, 500, 1000)


def summarize(sequences: list[list[tuple[int, int]]]) -> dict:
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=int)
    if lengths.size == 0:
        raise ValueError("Cannot profile an empty split.")
    return {
        "student_count": int(lengths.size),
        "history_length_unit": "interactions per student after the minimum-two-interaction inclusion rule",
        "minimum": int(lengths.min()),
        "maximum": int(lengths.max()),
        "mean": float(lengths.mean()),
        "quantiles": {f"p{int(100 * quantile):02d}": float(np.quantile(lengths, quantile)) for quantile in QUANTILES},
        "students_strictly_over_threshold": {
            str(threshold): {
                "count": int((lengths > threshold).sum()),
                "proportion": float((lengths > threshold).mean()),
            }
            for threshold in THRESHOLDS
        },
    }


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Controlled source data missing: {RAW_DATA}")
    split = SplitSpec()
    train, validation, test, labels, support = load_train_fitted_splits(RAW_DATA, split)
    payload = {
        "experiment": "revision5_history_length_distribution",
        "analysis_label": "Aggregate-only history-length profile for the pre-specified student-disjoint split; it is descriptive and does not alter the archived Revision 3 result.",
        "source_sha256": stable_sha256(RAW_DATA),
        "student_level_split": {"seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "inclusion_rule": "Students with fewer than two valid chronological interactions are excluded before splitting.",
        "train_fitted_label_support": support,
        "pre_specified_context_thresholds": list(THRESHOLDS),
        "split_history_length_summary": {
            "train": summarize(train),
            "validation": summarize(validation),
            "test": summarize(test),
        },
        "interpretation_limit": "These are aggregate descriptions of the fixed split. They do not by themselves quantify the performance impact of truncated training windows; that question is addressed separately by the prospective context-window sensitivity audit.",
        "privacy": "No user identifiers, split memberships, raw sequences, predictions or checkpoints are written. Only aggregate length statistics are emitted.",
    }
    encoded = json.dumps(payload, indent=2)
    ARCHIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    CONTROLLED_RESULTS.mkdir(parents=True, exist_ok=True)
    (ARCHIVE_RESULTS / "revision5_history_length_distribution.json").write_text(encoded, encoding="utf-8")
    (CONTROLLED_RESULTS / "revision5_history_length_distribution_private.json").write_text(encoded, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
