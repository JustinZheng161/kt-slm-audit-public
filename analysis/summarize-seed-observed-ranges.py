"""Report observed min–max dispersion from the archived per-seed aggregate records.

This is a descriptive reporting helper for Revision 5. It reports the literal
minimum and maximum of three observed seed-level values, alongside mean and sample
standard deviation. It does not estimate confidence intervals or perform hypothesis
or equivalence testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]


def result_directory(root: Path) -> Path:
    candidates = (
        root / "research_code" / "results",
        root / "results" / "revision3",
        root / "artifacts" / "revision3",
    )
    for candidate in candidates:
        if (candidate / "revision3-main-eight-epoch-probability-quality.json").exists():
            return candidate
    raise FileNotFoundError("No Revision 3 aggregate-result directory was found.")


RESULTS = result_directory(ROOT)


def describe(values: list[float]) -> dict:
    sample = np.asarray(values, dtype=float)
    if len(sample) < 2:
        raise ValueError("At least two seeds are required for a dispersion summary.")
    return {
        "seed_count": int(len(sample)),
        "per_seed_values": [float(value) for value in sample],
        "mean": float(sample.mean()),
        "standard_deviation": float(sample.std(ddof=1)),
        "observed_minimum": float(sample.min()),
        "observed_maximum": float(sample.max()),
        "observed_range_min_max": [float(sample.min()), float(sample.max())],
    }


def main() -> None:
    source = json.loads((RESULTS / "revision3-main-eight-epoch-probability-quality.json").read_text(encoding="utf-8"))
    output = {
        "experiment": "revision5_three_seed_observed_ranges",
        "analysis_scope": "Post hoc descriptive summary of the archived Revision 3 eight-epoch seed-level aggregate records.",
        "shared_evaluation_target": source["shared_evaluation_target"],
        "test_prediction_rows_per_seed": source["test_prediction_rows"],
        "configurations": {
            name: {metric: describe([row[metric] for row in rows]) for metric in ("roc_auc", "brier_score", "ece_10")}
            for name, rows in source["clean_dkt_runs"].items()
        },
        "interpretation_limit": "Each observed range is the literal minimum and maximum across three seed-level values. It is descriptive only and is not a confidence interval, bootstrap interval, hypothesis test or equivalence analysis.",
        "privacy": "The input and output contain only aggregate per-seed metrics; neither contains learner-level data, split memberships, predictions or checkpoints.",
    }
    target = RESULTS / "revision5-three-seed-observed-ranges.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
