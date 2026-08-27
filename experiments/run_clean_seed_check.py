"""Repeat the unregularized DKT baseline on the same three seeds used by AdamW.

This is a fairness check for the ablation: Adam and AdamW are compared with identical
data splits, architecture, learning rate, stopping rule, seeds, and test set. The
public output contains only aggregate metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_student_disjoint_kt import DKTSpec, RAW_DATA, SplitSpec, load_train_fitted_splits, metric_record, train_dkt


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if (_HERE.parents[2] / "research_code").exists() else _HERE.parents[1]
PUBLIC_RESULTS = ROOT / "research_code" / "results" if (ROOT / "research_code").exists() else ROOT / "results"


def main() -> None:
    torch.set_num_threads(4)
    train, validation, test, labels, _ = load_train_fitted_splits(RAW_DATA, SplitSpec())
    rows = []
    for seed in (20260822, 20260823, 20260824):
        spec = DKTSpec("DKT-64-Adam", 64, 64, 0.0, 0.002, 0.0, 8, seed)
        details, target, score, offsets = train_dkt(train, validation, test, len(labels), spec)
        record = metric_record(spec.name, target, score, offsets, seed=seed)
        record.update(details)
        rows.append(record)
    aucs = [row["roc_auc"] for row in rows]
    result = {
        "experiment": "clean_dkt64_adam_multiseed_fairness_check",
        "source_sha256": "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da",
        "student_level_split": {"seed": 20260822, "train_validation_test": [0.8, 0.1, 0.1]},
        "architecture": "embedding_dim=64, hidden_dim=64, dropout=0.0, learning_rate=0.002, epochs=8",
        "runs": rows,
        "aggregate": {
            "mean_roc_auc": float(np.mean(aucs)),
            "standard_deviation_roc_auc": float(np.std(aucs, ddof=1)),
            "seed_count": len(aucs),
        },
        "privacy": "No raw interactions, student IDs, memberships, or row-level predictions are stored in this public output.",
    }
    for destination in (RAW_DATA.parents[1] / "results" / "clean_dkt64_adam_multiseed_private.json", PUBLIC_RESULTS / "clean_dkt64_adam_multiseed.json"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
