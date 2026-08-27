"""Prospective sensitivity analysis for DKT training-window length.

The script is deliberately separate from the archived Revision 3 primary result.
It trains the same compact DKT-64 configuration under three pre-specified maximum
history windows (200, 500 and full available history) while evaluating complete
student histories. All other reported settings, including student split, seed set,
optimizer, learning rate, batch size, maximum epoch budget and validation-only
checkpoint selection, remain fixed. Only aggregate metrics are written.

For the full-history condition, batch size is held at 64 to preserve the optimizer
batching convention. If the controlled hardware cannot accommodate this condition,
the run must fail rather than silently substitute a different batch size; any
explicitly respecified memory-matched protocol should be reported separately.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_revision3_extended_evidence import enriched_metric, summarise_runs
from run_student_disjoint_kt import DKTSpec, RAW_DATA, SplitSpec, load_train_fitted_splits, stable_sha256, train_dkt


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]


def archive_result_directory(root: Path) -> Path:
    if (root / "research_code").exists():
        return root / "research_code" / "results"
    if (root / "results" / "revision3").exists():
        return root / "results" / "revision3"
    return root / "artifacts" / "revision3"


ARCHIVE_RESULTS = archive_result_directory(ROOT)
CONTROLLED_RESULTS = RAW_DATA.parents[1] / "results"
SEEDS = (20260822, 20260823, 20260824)
BATCH_SIZE = 64
EPOCH_CAP = 8


def run_condition(
    label: str,
    max_length: int,
    train: list,
    validation: list,
    test: list,
    skill_count: int,
) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        spec = DKTSpec(f"DKT-64-{label}", 64, 64, 0.0, 0.002, 0.0, EPOCH_CAP, seed)
        details, targets, scores, offsets = train_dkt(
            train,
            validation,
            test,
            skill_count,
            spec,
            max_length=max_length,
            evaluation_context_window=None,
            batch_size=BATCH_SIZE,
        )
        row = enriched_metric(spec.name, targets, scores, offsets, seed)
        row.update(details)
        rows.append(row)
    return rows


def paired_difference(reference: list[dict], candidate: list[dict], metric: str = "roc_auc") -> dict:
    reference_by_seed = {row["specification"]["seed"]: row for row in reference}
    candidate_by_seed = {row["specification"]["seed"]: row for row in candidate}
    if set(reference_by_seed) != set(candidate_by_seed):
        raise ValueError("Window sensitivity comparison requires identical seed sets.")
    values = np.asarray(
        [candidate_by_seed[seed][metric] - reference_by_seed[seed][metric] for seed in sorted(reference_by_seed)],
        dtype=float,
    )
    return {
        "candidate_minus_200_per_seed": [float(value) for value in values],
        "mean_difference": float(values.mean()),
        "standard_deviation_difference": float(values.std(ddof=1)),
    }


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Controlled source data missing: {RAW_DATA}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    split = SplitSpec()
    train, validation, test, labels, support = load_train_fitted_splits(RAW_DATA, split)
    full_history_window = max(len(sequence) - 1 for sequence in train)
    conditions = {
        "window_200": 200,
        "window_500": 500,
        "full_available_history": full_history_window,
    }
    runs = {
        label: run_condition(label, max_length, train, validation, test, len(labels))
        for label, max_length in conditions.items()
    }
    expected_targets = sum(len(sequence) - 1 for sequence in test)
    if {row["test_prediction_rows"] for rows in runs.values() for row in rows} != {expected_targets}:
        raise AssertionError("All window conditions must score the same full-history next-response target set.")
    payload = {
        "experiment": "revision5_training_window_length_sensitivity",
        "analysis_label": "Prospective sensitivity analysis; distinct from the archived Revision 3 primary result.",
        "source_sha256": stable_sha256(RAW_DATA),
        "student_level_split": {"seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "train_fitted_label_support": support,
        "seed_set": list(SEEDS),
        "shared_evaluation_target": "All second-and-later interactions of fixed test students, scored with complete preceding histories under every condition.",
        "fixed_settings": {"embedding_dim": 64, "hidden_dim": 64, "dropout": 0.0, "optimizer": "AdamW with weight_decay=0", "learning_rate": 0.002, "batch_size": BATCH_SIZE, "gradient_clip_norm": 5.0, "maximum_epochs": EPOCH_CAP, "checkpoint_selection": "highest validation ROC-AUC within the eight-epoch cap"},
        "training_window_conditions": conditions,
        "runs": runs,
        "aggregate": {label: summarise_runs(rows) for label, rows in runs.items()},
        "paired_roc_auc_difference_vs_window_200": {
            label: paired_difference(runs["window_200"], rows)
            for label, rows in runs.items() if label != "window_200"
        },
        "interpretation_limit": "This analysis isolates a training-window length choice under one source file, one student-disjoint split and an eight-epoch resource cap. It does not establish a global optimum, cross-dataset generality or a definitive null effect.",
        "privacy": "Only aggregate per-seed metrics and settings are written. Raw interactions, student IDs, split memberships, sequences, predictions and checkpoints remain in the controlled data root.",
    }
    encoded = json.dumps(payload, indent=2)
    ARCHIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    CONTROLLED_RESULTS.mkdir(parents=True, exist_ok=True)
    (ARCHIVE_RESULTS / "revision5-training-window-length-sensitivity.json").write_text(encoded, encoding="utf-8")
    (CONTROLLED_RESULTS / "revision5_training_window_length_sensitivity_private.json").write_text(encoded, encoding="utf-8")
    print(json.dumps({"aggregate": payload["aggregate"], "output": "revision5-training-window-length-sensitivity.json"}, indent=2))


if __name__ == "__main__":
    main()
