"""Audit whether DKT evaluation changes when history availability matches training chunks.

This experiment is intentionally separate from the archived Revision 3 result. It
compares two fully specified protocols that share the same student split, source
checksum, architecture, optimizer, seed set, eight-epoch budget and validation-only
model selection rule:

* ``full_student_history_legacy`` evaluates each test history as one sequence.
* ``matched_train_chunks`` evaluates the same non-overlapping 200-transition chunks
  used by training.

Only per-seed aggregate metrics and protocol metadata are written. Raw records,
student memberships, predictions and checkpoints remain in ``KT_AUDIT_DATA_ROOT``.
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
# Public layout: <repo>/experiments; private layout: <repo>/code/experiments.
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]


def archive_result_directory(root: Path) -> Path:
    """Select the tracked aggregate-only result directory for either repository layer."""
    if (root / "research_code").exists():
        return root / "research_code" / "results"
    if (root / "results" / "revision3").exists():
        return root / "results" / "revision3"
    return root / "artifacts" / "revision3"


ARCHIVE_RESULTS = archive_result_directory(ROOT)
CONTROLLED_RESULTS = RAW_DATA.parents[1] / "results"
SEEDS = (20260822, 20260823, 20260824)
MAX_HISTORY = 200


def run_protocol(
    name: str,
    evaluation_context_window: int | None,
    train: list,
    validation: list,
    test: list,
    skill_count: int,
) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        spec = DKTSpec(name, 64, 64, 0.0, 0.002, 0.0, 8, seed)
        details, target, score, offsets = train_dkt(
            train,
            validation,
            test,
            skill_count,
            spec,
            max_length=MAX_HISTORY,
            evaluation_context_window=evaluation_context_window,
        )
        row = enriched_metric(name, target, score, offsets, seed)
        row.update(details)
        rows.append(row)
    return rows


def paired_summary(reference: list[dict], candidate: list[dict]) -> dict:
    by_seed_reference = {row["specification"]["seed"]: row for row in reference}
    by_seed_candidate = {row["specification"]["seed"]: row for row in candidate}
    if set(by_seed_reference) != set(by_seed_candidate):
        raise ValueError("Context-parity audit requires identical seed sets.")
    summary: dict[str, dict] = {}
    for metric in ("roc_auc", "brier_score", "ece_10"):
        values = np.asarray(
            [by_seed_candidate[seed][metric] - by_seed_reference[seed][metric] for seed in sorted(by_seed_reference)],
            dtype=float,
        )
        summary[metric] = {
            "matched_chunks_minus_full_history_per_seed": [float(value) for value in values],
            "mean_difference": float(values.mean()),
            "standard_deviation_difference": float(values.std(ddof=1)),
        }
    return summary


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Controlled source data missing: {RAW_DATA}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    split = SplitSpec()
    train, validation, test, labels, support = load_train_fitted_splits(RAW_DATA, split)
    full_history = run_protocol("DKT-64-full-history", None, train, validation, test, len(labels))
    matched_chunks = run_protocol("DKT-64-matched-chunks", MAX_HISTORY, train, validation, test, len(labels))
    target_counts = {row["test_prediction_rows"] for row in [*full_history, *matched_chunks]}
    if target_counts != {sum(len(sequence) - 1 for sequence in test)}:
        raise AssertionError("Both context policies must score every shared next-response target exactly once.")

    payload = {
        "experiment": "revision4_context_parity_audit",
        "analysis_label": "Prospective context-availability sensitivity audit; not part of the archived Revision 3 primary result.",
        "source_sha256": stable_sha256(RAW_DATA),
        "student_level_split": {"seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "train_fitted_label_support": support,
        "architecture": {"embedding_dim": 64, "hidden_dim": 64, "dropout": 0.0},
        "optimization": {"optimizer": "AdamW with weight_decay=0", "learning_rate": 0.002, "gradient_clip_norm": 5.0, "maximum_epochs": 8},
        "sequence_window": {"training_max_transitions": MAX_HISTORY, "full_history_protocol": "One full student history at validation and test.", "matched_chunks_protocol": "Same non-overlapping 200-transition chunks at validation and test."},
        "seed_set": list(SEEDS),
        "shared_evaluation_target": "Every second-and-later interaction of the fixed test students; each target is counted exactly once under both policies.",
        "full_student_history_legacy_runs": full_history,
        "matched_train_chunks_runs": matched_chunks,
        "aggregate": {"full_student_history_legacy": summarise_runs(full_history), "matched_train_chunks": summarise_runs(matched_chunks)},
        "paired_matched_chunks_minus_full_history": paired_summary(full_history, matched_chunks),
        "interpretation_limit": "This audit isolates a context-availability protocol choice under one source file and split. It does not establish external generality or a ranking against other KT methods.",
        "privacy": "Only aggregate per-seed metrics and protocol metadata are written. Raw rows, student IDs, split memberships, sequences, predictions and checkpoints remain in the controlled data root.",
    }
    encoded = json.dumps(payload, indent=2)
    ARCHIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    CONTROLLED_RESULTS.mkdir(parents=True, exist_ok=True)
    (ARCHIVE_RESULTS / "revision4_context_parity_audit.json").write_text(encoded, encoding="utf-8")
    (CONTROLLED_RESULTS / "revision4_context_parity_audit_private.json").write_text(encoded, encoding="utf-8")
    print(json.dumps({"aggregate": payload["aggregate"], "output": "revision4_context_parity_audit.json"}, indent=2))


if __name__ == "__main__":
    main()
