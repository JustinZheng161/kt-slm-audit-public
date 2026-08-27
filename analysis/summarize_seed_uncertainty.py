"""Summarise existing per-seed metrics without accessing controlled learner records.

The output is a transparent descriptive appendix for the archived three-seed primary
analysis. Its t-reference ranges are not population confidence intervals, hypothesis
tests, or equivalence evidence; they make the sampling limitation visible alongside
mean and standard deviation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
# Public layout: <repo>/analysis; private layout: <repo>/code/analysis.
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]


def result_directory(root: Path) -> Path:
    """Support the public ``results/revision3`` and private archive layouts."""
    candidates = (
        root / "research_code" / "results",
        root / "results" / "revision3",
        root / "artifacts" / "revision3",
    )
    for candidate in candidates:
        if (candidate / "revision3_main_eight_epoch_probability_quality.json").exists():
            return candidate
    raise FileNotFoundError("No Revision 3 aggregate-result directory was found.")


RESULTS = result_directory(ROOT)

# Two-sided 95% t critical values for df 1–30, indexed by degrees of freedom.
T_975 = (
    None,
    12.7062047364, 4.3026527297, 3.1824463053, 2.7764451052,
    2.5705818356, 2.4469118511, 2.3646242510, 2.3060041350,
    2.2621571629, 2.2281388520, 2.2009851601, 2.1788128297,
    2.1603686565, 2.1447866879, 2.1314495456, 2.1199052992,
    2.1098155778, 2.1009220402, 2.0930240544, 2.0859634473,
    2.0796138447, 2.0738730679, 2.0686576104, 2.0638985616,
    2.0595385528, 2.0555294386, 2.0518305165, 2.0484071418,
    2.0452296421, 2.0422724563,
)


def describe(values: list[float]) -> dict:
    sample = np.asarray(values, dtype=float)
    count = len(sample)
    if count < 2:
        raise ValueError("At least two seeds are required for a descriptive dispersion summary.")
    sd = float(sample.std(ddof=1))
    mean = float(sample.mean())
    standard_error = sd / math.sqrt(count)
    if count - 1 >= len(T_975):
        raise ValueError("Add a validated critical value before using more than 31 seeds.")
    half_width = T_975[count - 1] * standard_error
    return {
        "seed_count": count,
        "per_seed_values": [float(value) for value in sample],
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": standard_error,
        "descriptive_t_reference_range_95": [mean - half_width, mean + half_width],
        "minimum": float(sample.min()),
        "maximum": float(sample.max()),
    }


def main() -> None:
    main_result = json.loads((RESULTS / "revision3_main_eight_epoch_probability_quality.json").read_text(encoding="utf-8"))
    report = {
        "experiment": "revision4_descriptive_seed_uncertainty",
        "analysis_scope": "Post hoc descriptive summary of the archived Revision 3 eight-epoch per-seed aggregate records.",
        "shared_evaluation_target": main_result["shared_evaluation_target"],
        "test_prediction_rows_per_seed": main_result["test_prediction_rows"],
        "configurations": {},
        "interpretation_limit": "The displayed t-reference ranges summarize only three observed initialization seeds. They are not learner-cluster bootstrap intervals, population confidence intervals, significance tests, equivalence tests, or evidence that any two configurations are the same.",
        "privacy": "The input and output contain only aggregate per-seed metrics; neither includes learner-level records, split memberships, predictions or checkpoints.",
    }
    for name, rows in main_result["clean_dkt_runs"].items():
        report["configurations"][name] = {
            metric: describe([row[metric] for row in rows])
            for metric in ("roc_auc", "brier_score", "ece_10")
        }
    target = RESULTS / "revision4_descriptive_seed_uncertainty.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
