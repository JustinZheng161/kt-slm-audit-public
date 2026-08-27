"""Revision 6 aggregate-safe statistical and BKT diagnostics.

The script reads the permitted corrected ASSISTments source only from
``KT_AUDIT_DATA_ROOT``. It never publishes raw interactions, student IDs,
per-skill parameters, split membership, individual predictions, bootstrap
replicates, or model checkpoints. Public output contains only aggregate
summaries required for reviewer-facing reproducibility.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]
EXPERIMENTS = ROOT / "research_code" / "experiments" if (ROOT / "research_code").exists() else ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_student_disjoint_kt import (  # noqa: E402
    DKTSpec,
    RAW_DATA,
    SplitSpec,
    bkt_em,
    bkt_predictions,
    evaluate_skill_prior,
    load_train_fitted_splits,
    per_skill_response_sequences,
    set_seed,
    stable_sha256,
    train_dkt,
)

RESULTS = ROOT / "research_code" / "results" / "revision3" if (ROOT / "research_code").exists() else ROOT / "results" / "revision3"
CONTROLLED_ROOT = Path(os.environ.get("KT_AUDIT_DATA_ROOT", ROOT / "private_data"))
CONTROLLED_RESULTS = CONTROLLED_ROOT / "results"


def clean(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def observed_sd_mde_reference(sd: float, sample_size: int, alpha: float = 0.05, power: float = 0.80) -> float | None:
    """Return a transparent t-quantile design-sensitivity reference.

    The expression ``(t_(1-alpha/2,df) + t_(power,df)) * SD / sqrt(n)`` is an
    approximate fixed-n two-sided paired-test planning reference. It is used
    only to put the n=3 observed differences on a numerical scale; it is not
    an exact retrospective power calculation, a pre-specified SESOI or an
    equivalence margin.
    """
    if not np.isfinite(sd) or sd <= 0.0 or sample_size < 2:
        return None
    degrees_of_freedom = sample_size - 1
    multiplier = t.ppf(1.0 - alpha / 2.0, degrees_of_freedom) + t.ppf(power, degrees_of_freedom)
    return float(multiplier * sd / np.sqrt(sample_size))


def paired_effect_summary() -> dict:
    payload = json.loads((RESULTS / "revision3_descriptive_paired_metric_differences.json").read_text(encoding="utf-8"))
    main = json.loads((RESULTS / "revision3_main_eight_epoch_probability_quality.json").read_text(encoding="utf-8"))
    reference_sd = float(main["clean_dkt_aggregate"]["DKT-64-Adam"]["standard_deviation_roc_auc"])
    result: dict[str, dict] = {}
    for key, comparison in payload["comparisons"].items():
        differences = np.asarray(comparison["metrics"]["roc_auc"]["candidate_minus_reference_per_seed"], dtype=float)
        sample_size = len(differences)
        mean_difference = float(differences.mean())
        paired_sd = float(differences.std(ddof=1)) if sample_size > 1 else float("nan")
        result[key] = {
            "comparison": comparison["comparison"],
            "paired_seed_count": sample_size,
            "mean_auc_difference": mean_difference,
            "absolute_mean_difference": abs(mean_difference),
            "paired_difference_standard_deviation": clean(paired_sd),
            "paired_standardized_mean_difference_cohens_dz": clean(mean_difference / paired_sd) if paired_sd > 0 else None,
            "reference_condition_cross_seed_standard_deviation": reference_sd,
            "absolute_mean_difference_divided_by_reference_cross_seed_sd": abs(mean_difference) / reference_sd if reference_sd > 0 else None,
            "observed_sd_power_reference": {
                "assumed_test": "two-sided paired t reference",
                "alpha": 0.05,
                "target_power": 0.80,
                "minimum_detectable_auc_difference_reference": observed_sd_mde_reference(paired_sd, sample_size),
                "calculation": "(t_(1-alpha/2, df) + t_(power, df)) * paired-SD / sqrt(n), with df=n-1",
                "interpretation": "Post hoc observed-SD planning reference only; it is neither an exact retrospective power calculation, a pre-specified SESOI nor an equivalence margin and cannot support a no-effect conclusion.",
            },
        }
    return {
        "experiment": "revision6_standardized_paired_effect_summary",
        "analysis_scope": "Primary eight-epoch DKT comparisons; same three seeds and 24,306 test targets per seed.",
        "reference_condition": "DKT-64 Adam",
        "comparisons": result,
        "interpretation_limit": "Cohen's dz uses the standard deviation of the three paired seed differences. With n=3 it is descriptive and unstable. MDE references are calculated from observed paired SD using a transparent t-quantile planning approximation after observing the data; they are not an exact retrospective power calculation, a SESOI, power guarantee, equivalence interval or TOST.",
        "privacy": "Aggregate seed-level metrics only; no student-level scores, predictions or bootstrap replicates are released.",
    }


def bkt_aggregate_diagnostics(train_sequences: list[list[tuple[int, int]]], skill_count: int) -> dict:
    by_skill = per_skill_response_sequences(train_sequences)
    parameter_rows: list[np.ndarray] = []
    final_changes: list[float] = []
    fallback_count = 0
    per_skill_observation_counts: list[int] = []
    for skill in range(skill_count):
        sequences = by_skill.get(skill, [])
        observation_count = sum(len(sequence) for sequence in sequences)
        per_skill_observation_counts.append(observation_count)
        if observation_count < 20:
            fallback_count += 1
        params_19 = bkt_em(sequences, iterations=19)
        params_20 = bkt_em(sequences, iterations=20)
        parameter_rows.append(params_20)
        final_changes.append(float(np.max(np.abs(params_20 - params_19))))
    parameters = np.vstack(parameter_rows)
    changes = np.asarray(final_changes, dtype=float)
    lower = np.asarray([0.01, 0.001, 0.001, 0.001])
    upper = np.asarray([0.99, 0.50, 0.49, 0.49])
    parameter_names = ("P(L0)", "P(T)", "P(G)", "P(S)")
    parameter_quantiles = {name: quantile_summary(parameters[:, index]) for index, name in enumerate(parameter_names)}
    boundary_counts = {
        name: {
            "within_0.01_of_lower_bound": int(np.sum(parameters[:, index] <= lower[index] + 0.01)),
            "within_0.01_of_upper_bound": int(np.sum(parameters[:, index] >= upper[index] - 0.01)),
        }
        for index, name in enumerate(parameter_names)
    }
    return {
        "experiment": "revision6_bkt_per_skill_aggregate_diagnostics",
        "fit_scope": "Independent two-state BKT EM per train-fitted atomic skill; fixed 20 iterations.",
        "skill_count": int(skill_count),
        "short_data_fallback_skill_count": int(fallback_count),
        "training_observations_per_skill": quantile_summary(np.asarray(per_skill_observation_counts, dtype=float)),
        "parameter_quantiles_across_skills": parameter_quantiles,
        "parameter_bound_proximity_counts": boundary_counts,
        "final_iteration_max_absolute_parameter_change": {
            "threshold_for_parameter_stability_proxy": 1e-5,
            "skills_at_or_below_threshold": int(np.sum(changes <= 1e-5)),
            "proportion_at_or_below_threshold": float(np.mean(changes <= 1e-5)),
            "quantiles": quantile_summary(changes),
        },
        "interpretation_limit": "The iteration-19 to iteration-20 parameter change is a fixed-iteration stability proxy, not a likelihood-based convergence certificate, a global-optimum proof or a multi-start diagnostic. Per-skill parameter values are intentionally not released.",
        "privacy": "Only cross-skill aggregate quantiles and counts are released; no skill identifiers, per-skill parameter vectors, learners or interactions are released.",
    }


def bkt_extended_iteration_sensitivity(
    train_sequences: list[list[tuple[int, int]]],
    test_sequences: list[list[tuple[int, int]]],
    skill_count: int,
) -> dict:
    """Compare 20 versus 100 fixed EM iterations using aggregate-only output."""
    grouped = per_skill_response_sequences(train_sequences)
    parameters_20 = np.stack([bkt_em(grouped.get(skill, []), iterations=20) for skill in range(skill_count)])
    parameters_99 = np.stack([bkt_em(grouped.get(skill, []), iterations=99) for skill in range(skill_count)])
    parameters_100 = np.stack([bkt_em(grouped.get(skill, []), iterations=100) for skill in range(skill_count)])
    targets_20, scores_20, _ = bkt_predictions(test_sequences, parameters_20)
    targets_100, scores_100, _ = bkt_predictions(test_sequences, parameters_100)
    final_changes_100 = np.max(np.abs(parameters_100 - parameters_99), axis=1)
    threshold = 1e-5
    return {
        "experiment": "revision6_bkt_fixed_iteration_sensitivity",
        "comparison": "Per-skill BKT with 20 versus 100 fixed EM iterations on the same train/test protocol.",
        "skill_count": int(skill_count),
        "shared_test_targets": int(len(targets_20)),
        "fixed_20_test_roc_auc": float(roc_auc_score(targets_20, scores_20)),
        "fixed_100_test_roc_auc": float(roc_auc_score(targets_100, scores_100)),
        "auc_difference_100_minus_20": float(roc_auc_score(targets_100, scores_100) - roc_auc_score(targets_20, scores_20)),
        "fixed_100_final_iteration_max_absolute_parameter_change": {
            "threshold_for_parameter_stability_proxy": threshold,
            "skills_at_or_below_threshold": int(np.sum(final_changes_100 <= threshold)),
            "proportion_at_or_below_threshold": float(np.mean(final_changes_100 <= threshold)),
            "quantiles": quantile_summary(final_changes_100),
        },
        "interpretation_limit": "The 100-iteration comparison is a post hoc implementation sensitivity check. Its final-update parameter change remains a stability proxy, not a likelihood-based convergence certificate, a multi-start analysis or an independently pre-specified benchmark.",
        "privacy": "Only cross-skill aggregate stability summaries and aggregate test ROC-AUC values are released; no per-skill parameters, learners, predictions or replicate values are released.",
    }


def paired_student_bootstrap_dkt_vs_bkt(
    train_sequences: list[list[tuple[int, int]]],
    validation_sequences: list[list[tuple[int, int]]],
    test_sequences: list[list[tuple[int, int]]],
    skill_count: int,
    bootstrap_replicates: int = 1000,
) -> dict:
    grouped = per_skill_response_sequences(train_sequences)
    parameters = np.stack([bkt_em(grouped.get(skill, []), iterations=20) for skill in range(skill_count)])
    bkt_targets, bkt_scores, bkt_offsets = bkt_predictions(test_sequences, parameters)
    specification = DKTSpec("DKT-64-Adam", 64, 64, 0.0, 0.002, 0.0, 8, 20260822)
    set_seed(specification.seed)
    _, dkt_targets, dkt_scores, dkt_offsets = train_dkt(
        train_sequences,
        validation_sequences,
        test_sequences,
        skill_count,
        specification,
        early_stop_patience=None,
        max_length=200,
        evaluation_context_window=None,
        batch_size=64,
    )
    if not np.array_equal(bkt_targets, dkt_targets) or bkt_offsets != dkt_offsets:
        raise RuntimeError("DKT and BKT test targets/offsets are not aligned for paired bootstrap.")
    rng = np.random.default_rng(20260826)
    differences = np.empty(bootstrap_replicates, dtype=float)
    offsets = dkt_offsets
    student_count = len(offsets)
    for index in range(bootstrap_replicates):
        sampled_students = rng.integers(0, student_count, size=student_count)
        sampled_rows = np.concatenate([np.arange(offsets[student][0], offsets[student][1]) for student in sampled_students])
        differences[index] = roc_auc_score(dkt_targets[sampled_rows], dkt_scores[sampled_rows]) - roc_auc_score(bkt_targets[sampled_rows], bkt_scores[sampled_rows])
    observed_difference = float(roc_auc_score(dkt_targets, dkt_scores) - roc_auc_score(bkt_targets, bkt_scores))
    probability_nonpositive = float(np.mean(differences <= 0.0))
    probability_nonnegative = float(np.mean(differences >= 0.0))
    return {
        "experiment": "revision6_student_cluster_paired_bootstrap_dkt64adam_minus_bkt",
        "comparison": "DKT-64 Adam seed 20260822 minus per-skill BKT EM",
        "shared_test_targets": int(len(dkt_targets)),
        "student_clusters": int(student_count),
        "bootstrap_replicates": int(bootstrap_replicates),
        "random_seed": 20260826,
        "observed_auc_difference": observed_difference,
        "percentile_95_confidence_interval": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "bootstrap_difference_summary": {"mean": float(differences.mean()), "standard_deviation": float(differences.std(ddof=1))},
        "two_sided_direction_reversal_proportion": float(min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))),
        "interpretation_limit": "This is a student-cluster paired bootstrap for the shared target set and one DKT seed versus one fixed-iteration BKT implementation. The direction-reversal proportion is reported as a descriptive bootstrap tail proportion, not a universal model-selection claim or cross-dataset inference.",
        "privacy": "Only aggregate bootstrap summary statistics are released; bootstrap replicate values, predictions, per-student AUCs and student membership remain controlled.",
    }


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Controlled source data missing: {RAW_DATA}")
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    CONTROLLED_RESULTS.mkdir(parents=True, exist_ok=True)
    paired_effects = paired_effect_summary()
    train, validation, test, labels, _ = load_train_fitted_splits(RAW_DATA, SplitSpec())
    bkt_diagnostics = bkt_aggregate_diagnostics(train, len(labels))
    bkt_iteration_sensitivity = bkt_extended_iteration_sensitivity(train, test, len(labels))
    dkt_bkt_bootstrap = paired_student_bootstrap_dkt_vs_bkt(train, validation, test, len(labels))
    common = {
        "source_sha256": stable_sha256(RAW_DATA),
        "source_file": RAW_DATA.name,
        "split_seed": 20260822,
        "privacy": "The public file contains only aggregate-safe statistics. Detailed source records, student membership, predictions, per-skill values and checkpoints remain in KT_AUDIT_DATA_ROOT.",
    }
    payload = {"provenance": common, "paired_effects": paired_effects, "bkt_diagnostics": bkt_diagnostics, "bkt_iteration_sensitivity": bkt_iteration_sensitivity, "dkt_bkt_paired_bootstrap": dkt_bkt_bootstrap}
    (RESULTS / "revision6_statistical_and_bkt_audits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    controlled_summary = {"provenance": common, "public_result": "revision6_statistical_and_bkt_audits.json", "status": "completed; no private raw outputs written by this script"}
    (CONTROLLED_RESULTS / "revision6_statistical_and_bkt_audits_private.json").write_text(json.dumps(controlled_summary, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
