"""Revision 3 aggregate-only evidence generation.

This script keeps all learner-level records and predictions in memory only. It writes
only aggregate metrics, calibration-bin summaries, and documented training histories.
The controlled ASSISTments source, split memberships, and any individual predictions
remain outside the repository under KT_AUDIT_DATA_ROOT.

The analysis has three deliberately separated components:
1. Main eight-epoch comparison, including probability-quality metrics.
2. Exploratory extension with a 20-epoch validation selection budget. Test data are
   evaluated exactly once after validation selection for each seed.
3. Synthetic training-label inversion sensitivity at 5%, 10%, and 20%.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_label_noise_robustness import corrupt_training_labels
from run_student_disjoint_kt import (
    DKTSpec,
    RAW_DATA,
    SplitSpec,
    bkt_predictions,
    evaluate_skill_prior,
    fit_bkt_models,
    load_train_fitted_splits,
    make_skill_priors,
    metric_record,
    stable_sha256,
    train_dkt,
)


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if (HERE.parents[2] / "research_code").exists() else HERE.parents[1]
PUBLIC_RESULTS = ROOT / "research_code" / "results" if (ROOT / "research_code").exists() else ROOT / "results"
PRIVATE_RESULTS = RAW_DATA.parents[1] / "results"
SEEDS = (20260822, 20260823, 20260824)
CALIBRATION_BIN_COUNT = 10


def probability_metrics(target: np.ndarray, score: np.ndarray) -> dict:
    """Return aggregate Brier/ECE and fixed-width reliability bins; retain no rows."""
    target = np.asarray(target, dtype=float)
    score = np.asarray(score, dtype=float)
    if target.size == 0 or target.size != score.size:
        raise ValueError("Probability metric input must be non-empty and aligned.")
    brier = float(np.mean((score - target) ** 2))
    bin_index = np.minimum((score * CALIBRATION_BIN_COUNT).astype(int), CALIBRATION_BIN_COUNT - 1)
    ece = 0.0
    bins: list[dict] = []
    for index in range(CALIBRATION_BIN_COUNT):
        mask = bin_index == index
        count = int(mask.sum())
        if count:
            mean_prediction = float(score[mask].mean())
            empirical_accuracy = float(target[mask].mean())
            ece += abs(empirical_accuracy - mean_prediction) * count / target.size
        else:
            mean_prediction = None
            empirical_accuracy = None
        bins.append(
            {
                "bin": index,
                "lower_inclusive": index / CALIBRATION_BIN_COUNT,
                "upper_exclusive": (index + 1) / CALIBRATION_BIN_COUNT,
                "count": count,
                "mean_prediction": mean_prediction,
                "empirical_accuracy": empirical_accuracy,
            }
        )
    return {"brier_score": brier, "ece_10": float(ece), "calibration_bins_10": bins}


def enriched_metric(name: str, target: np.ndarray, score: np.ndarray, offsets: list[tuple[int, int]], seed: int) -> dict:
    record = metric_record(name, target, score, offsets, seed=seed)
    record.update(probability_metrics(target, score))
    return record


def summarise_runs(rows: list[dict]) -> dict:
    """Aggregate equal-size per-seed test runs without retaining their predictions."""
    if not rows:
        raise ValueError("Cannot summarise an empty run collection.")
    prediction_counts = {row["test_prediction_rows"] for row in rows}
    if len(prediction_counts) != 1:
        raise ValueError(f"Mismatched score target counts: {prediction_counts}")
    output = {
        "seed_count": len(rows),
        "test_prediction_rows_per_seed": int(next(iter(prediction_counts))),
        "mean_roc_auc": float(np.mean([row["roc_auc"] for row in rows])),
        "standard_deviation_roc_auc": float(np.std([row["roc_auc"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "mean_brier_score": float(np.mean([row["brier_score"] for row in rows])),
        "standard_deviation_brier_score": float(np.std([row["brier_score"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "mean_ece_10": float(np.mean([row["ece_10"] for row in rows])),
        "standard_deviation_ece_10": float(np.std([row["ece_10"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
    }
    mean_bins = []
    for bin_index in range(CALIBRATION_BIN_COUNT):
        observed = [row["calibration_bins_10"][bin_index] for row in rows]
        nonempty = [entry for entry in observed if entry["count"] > 0]
        mean_bins.append(
            {
                "bin": bin_index,
                "lower_inclusive": bin_index / CALIBRATION_BIN_COUNT,
                "upper_exclusive": (bin_index + 1) / CALIBRATION_BIN_COUNT,
                "mean_count_across_seeds": float(np.mean([entry["count"] for entry in observed])),
                "mean_prediction_across_seeds": float(np.mean([entry["mean_prediction"] for entry in nonempty])) if nonempty else None,
                "empirical_accuracy_across_seeds": float(np.mean([entry["empirical_accuracy"] for entry in nonempty])) if nonempty else None,
            }
        )
    output["mean_calibration_bins_10"] = mean_bins
    return output


def write_aggregate(name: str, payload: dict) -> None:
    PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)
    PRIVATE_RESULTS.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2)
    (PUBLIC_RESULTS / name).write_text(encoded, encoding="utf-8")
    (PRIVATE_RESULTS / name.replace(".json", "_private.json")).write_text(encoded, encoding="utf-8")


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Controlled source data missing: {RAW_DATA}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    split = SplitSpec()
    train, validation, test, labels, support = load_train_fitted_splits(RAW_DATA, split)
    source_sha = stable_sha256(RAW_DATA)

    # Main analysis: the original eight-epoch resource-bound protocol.
    prior = make_skill_priors(train, len(labels))
    prior_target, prior_score, prior_offsets = evaluate_skill_prior(test, prior)
    bkt_parameters = fit_bkt_models(train, len(labels))
    bkt_target, bkt_score, bkt_offsets = bkt_predictions(test, bkt_parameters)
    single_model_rows = {
        "Skill-prior": enriched_metric("Skill-prior", prior_target, prior_score, prior_offsets, split.seed),
        "BKT-per-skill-EM": enriched_metric("BKT-per-skill-EM", bkt_target, bkt_score, bkt_offsets, split.seed + 1),
    }
    clean_specs = {
        "DKT-64-Adam": (64, 64, 0.0, 0.002, 0.0),
        "DKT-64-AdamW": (64, 64, 0.0, 0.002, 0.0001),
        "DKT-96-AdamW-dropout": (96, 96, 0.10, 0.001, 0.0001),
    }
    clean_runs: dict[str, list[dict]] = defaultdict(list)
    for name, (embedding, hidden, dropout, learning_rate, weight_decay) in clean_specs.items():
        for seed in SEEDS:
            spec = DKTSpec(name, embedding, hidden, dropout, learning_rate, weight_decay, 8, seed)
            details, target, score, offsets = train_dkt(train, validation, test, len(labels), spec)
            row = enriched_metric(name, target, score, offsets, seed)
            row.update(details)
            clean_runs[name].append(row)
    if any(row["test_prediction_rows"] != len(prior_target) for rows in clean_runs.values() for row in rows):
        raise AssertionError("Main DKT target count does not match the baseline target count.")
    main_result = {
        "experiment": "revision3_main_eight_epoch_probability_quality",
        "analysis_label": "Primary analysis conditional on the pre-specified eight-epoch computational budget.",
        "source_sha256": source_sha,
        "student_level_split": {"seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "train_fitted_label_support": support,
        "shared_evaluation_target": "second and later interaction for each test student",
        "test_prediction_rows": int(len(prior_target)),
        "metrics": {
            "roc_auc": "Discrimination over the shared test target set.",
            "brier_score": "Mean squared error of predicted correctness probabilities; lower is better.",
            "ece_10": "Expected calibration error with ten fixed-width [0,1] probability bins; lower is better. It is descriptive, not a statistical test.",
        },
        "single_run_references": single_model_rows,
        "clean_dkt_runs": clean_runs,
        "clean_dkt_aggregate": {name: summarise_runs(rows) for name, rows in clean_runs.items()},
        "bkt_em_implementation": {
            "model": "Independent two-state BKT per skill.",
            "initial_parameters": {"P_L0": 0.25, "P_T": 0.12, "P_G": 0.20, "P_S": 0.10},
            "iterations": 20,
            "fixed_iterations_not_convergence_test": True,
            "parameter_bounds": {"P_L0": [0.01, 0.99], "P_T": [0.001, 0.50], "P_G": [0.001, 0.49], "P_S": [0.001, 0.49]},
            "short_sequence_fallback": {"threshold_total_observations": 20, "parameters": [0.25, 0.10, 0.20, 0.10]},
            "interpretation_limit": "No multi-start sensitivity analysis or per-skill convergence diagnostic is claimed; BKT serves as a transparent reference baseline.",
        },
        "privacy": "Only aggregate metrics and bins are written. Raw rows, identifiers, split memberships, and individual predictions remain in the controlled data root.",
    }
    write_aggregate("revision3-main-eight-epoch-probability-quality.json", main_result)

    # Exploratory extension: the twenty-epoch cap was motivated by a later diagnostic.
    extended_runs: list[dict] = []
    for seed in SEEDS:
        spec = DKTSpec("DKT-64-Adam-exploratory-20epoch", 64, 64, 0.0, 0.002, 0.0, 20, seed)
        details, target, score, offsets = train_dkt(
            train, validation, test, len(labels), spec, record_training_auc=True, early_stop_patience=None
        )
        row = enriched_metric(spec.name, target, score, offsets, seed)
        row.update(details)
        extended_runs.append(row)
    if any(row["test_prediction_rows"] != len(prior_target) for row in extended_runs):
        raise AssertionError("Extended DKT target count does not match the baseline target count.")
    extended_result = {
        "experiment": "revision3_exploratory_extended_budget",
        "analysis_label": "Post hoc exploratory extension. For each seed, the model is trained for all 20 epochs; the checkpoint with the maximum validation AUC is selected before its single test evaluation.",
        "source_sha256": source_sha,
        "student_level_split": {"seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "test_prediction_rows_per_seed": int(len(prior_target)),
        "runs": extended_runs,
        "aggregate": summarise_runs(extended_runs),
        "interpretation_limit": "This is not a replacement for the pre-specified eight-epoch main analysis and it is not a new externally validated benchmark claim. It reports the consequence of extending the validation-selection budget after the original analysis.",
        "privacy": "Only aggregate metrics, selected epochs, and aggregate training histories are written.",
    }
    write_aggregate("revision3-exploratory-extended-budget.json", extended_result)

    # Sensitivity curve: rates are fixed before these reruns and never applied to val/test.
    noise_rates = (0.05, 0.10, 0.20)
    noise_runs: list[dict] = []
    for rate in noise_rates:
        for name, (embedding, hidden, dropout, learning_rate, weight_decay) in clean_specs.items():
            if name == "DKT-96-AdamW-dropout":
                continue
            for seed in SEEDS:
                noisy_train, flips = corrupt_training_labels(train, rate, seed=seed + 1000)
                spec = DKTSpec(name, embedding, hidden, dropout, learning_rate, weight_decay, 8, seed)
                details, target, score, offsets = train_dkt(noisy_train, validation, test, len(labels), spec)
                row = enriched_metric(name, target, score, offsets, seed)
                row.update(details)
                row["label_noise_rate"] = rate
                row["training_label_flips"] = flips
                noise_runs.append(row)
    if any(row["test_prediction_rows"] != len(prior_target) for row in noise_runs):
        raise AssertionError("Noise-sensitivity DKT target count does not match the baseline target count.")
    by_rate_and_method: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in noise_runs:
        by_rate_and_method[f"{row['label_noise_rate']:.2f}"][row["name"]].append(row)
    sensitivity_summary = {
        rate: {name: summarise_runs(rows) for name, rows in methods.items()}
        for rate, methods in by_rate_and_method.items()
    }
    noise_result = {
        "experiment": "revision3_training_label_inversion_sensitivity",
        "analysis_label": "Synthetic sensitivity analysis, not a behavioral robustness claim.",
        "source_sha256": source_sha,
        "student_level_split": {"seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "test_prediction_rows_per_seed": int(len(prior_target)),
        "perturbation": {
            "type": "independent binary outcome inversion in training data only",
            "rates": list(noise_rates),
            "validation_and_test_unchanged": True,
            "generator_seed_rule": "each model seed plus 1000",
        },
        "runs": noise_runs,
        "aggregate_by_rate_and_method": sensitivity_summary,
        "clean_reference_from_same_script": {
            name: summarise_runs(rows) for name, rows in clean_runs.items() if name in ("DKT-64-Adam", "DKT-64-AdamW")
        },
        "privacy": "Only aggregate metrics, counts, and calibration bins are written; no interaction-level data or predictions are emitted.",
    }
    write_aggregate("revision3-training-label-inversion-sensitivity.json", noise_result)

    print(
        json.dumps(
            {
                "main": {name: values["mean_roc_auc"] for name, values in main_result["clean_dkt_aggregate"].items()},
                "extended": extended_result["aggregate"]["mean_roc_auc"],
                "noise_rates": list(sensitivity_summary),
                "outputs": [
                    "revision3-main-eight-epoch-probability-quality.json",
                    "revision3-exploratory-extended-budget.json",
                    "revision3-training-label-inversion-sensitivity.json",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
