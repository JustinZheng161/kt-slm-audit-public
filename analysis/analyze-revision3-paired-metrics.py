"""Describe paired three-seed differences from Revision 3 aggregate evidence.

The script reads aggregate per-seed metric records only and writes aggregate
differences. Its empirical seed-resampling intervals quantify observed variation
with n=3; they are explicitly not equivalence tests, confidence intervals for a
population effect, or evidence of no effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if (HERE.parents[2] / "research_code").exists() else HERE.parents[1]
RESULTS = ROOT / "research_code" / "results" if (ROOT / "research_code").exists() else ROOT / "results"


def interval(values: np.ndarray, seed: int, replicates: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled_means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return [float(np.quantile(sampled_means, 0.025)), float(np.quantile(sampled_means, 0.975))]


def compare(reference_rows: list[dict], candidate_rows: list[dict], label: str, seed: int) -> dict:
    reference_by_seed = {row["specification"]["seed"]: row for row in reference_rows}
    candidate_by_seed = {row["specification"]["seed"]: row for row in candidate_rows}
    if set(reference_by_seed) != set(candidate_by_seed):
        raise ValueError(f"{label}: paired seed mismatch.")
    metrics = {}
    for offset, metric in enumerate(("roc_auc", "brier_score", "ece_10")):
        differences = np.asarray(
            [candidate_by_seed[key][metric] - reference_by_seed[key][metric] for key in sorted(reference_by_seed)],
            dtype=float,
        )
        metrics[metric] = {
            "candidate_minus_reference_per_seed": [float(value) for value in differences],
            "mean_difference": float(differences.mean()),
            "descriptive_seed_resampling_95_interval": interval(differences, seed + offset),
        }
    return {"comparison": label, "paired_seed_count": len(reference_by_seed), "metrics": metrics}


def main() -> None:
    main_result = json.loads((RESULTS / "revision3-main-eight-epoch-probability-quality.json").read_text(encoding="utf-8"))
    runs = main_result["clean_dkt_runs"]
    output = {
        "experiment": "revision3_descriptive_paired_metric_differences",
        "analysis_scope": "Primary eight-epoch analysis only; each comparison uses the same three seeds and 24,306 targets per seed.",
        "comparisons": {
            "DKT-64-AdamW_minus_DKT-64-Adam": compare(runs["DKT-64-Adam"], runs["DKT-64-AdamW"], "DKT-64 AdamW − DKT-64 Adam", 20260830),
            "DKT-96-AdamW-dropout_minus_DKT-64-Adam": compare(runs["DKT-64-Adam"], runs["DKT-96-AdamW-dropout"], "DKT-96 AdamW + dropout − DKT-64 Adam", 20260901),
        },
        "interpretation_limit": "The n=3 seed-resampling intervals describe the observed configuration-specific variation. They do not implement TOST, do not use a pre-specified equivalence margin, and must not be read as evidence of no effect or practical equivalence.",
        "privacy": "The output contains only aggregate per-seed metric differences; it includes no learner-level data or predictions.",
    }
    target = RESULTS / "revision3-descriptive-paired-metric-differences.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
