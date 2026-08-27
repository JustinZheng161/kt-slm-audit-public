"""Evaluate DKT configurations under controlled training-label perturbation.

The test and validation students are never perturbed. Only a seeded fraction of
training outcomes is inverted. This is a protocol-level robustness check; it does
not identify real learner errors or make claims about educational deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_student_disjoint_kt import (
    DKTSpec,
    RAW_DATA,
    SplitSpec,
    load_train_fitted_splits,
    metric_record,
    split_students,
    train_dkt,
)


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if (_HERE.parents[2] / "research_code").exists() else _HERE.parents[1]
PRIVATE_DIR = RAW_DATA.parents[1] / "results"
PUBLIC_DIR = ROOT / "research_code" / "results" if (ROOT / "research_code").exists() else ROOT / "results"


def corrupt_training_labels(sequences: list[list[tuple[int, int]]], rate: float, seed: int) -> tuple[list[list[tuple[int, int]]], int]:
    rng = np.random.default_rng(seed)
    corrupted: list[list[tuple[int, int]]] = []
    flips = 0
    for sequence in sequences:
        replacement: list[tuple[int, int]] = []
        for skill, label in sequence:
            if rng.random() < rate:
                replacement.append((skill, 1 - label))
                flips += 1
            else:
                replacement.append((skill, label))
        corrupted.append(replacement)
    return corrupted, flips


def main() -> None:
    torch.set_num_threads(4)
    rate = 0.10
    split = SplitSpec()
    train, validation, test, labels, _ = load_train_fitted_splits(RAW_DATA, split)
    specs = [
        DKTSpec("DKT-64-Adam", 64, 64, 0.0, 0.002, 0.0, 8, seed)
        for seed in (20260822, 20260823, 20260824)
    ] + [
        DKTSpec("DKT-64-AdamW", 64, 64, 0.0, 0.002, 0.0001, 8, seed)
        for seed in (20260822, 20260823, 20260824)
    ]
    outputs = []
    for spec in specs:
        noisy_train, flip_count = corrupt_training_labels(train, rate, seed=spec.seed + 1000)
        details, target, score, offsets = train_dkt(noisy_train, validation, test, len(labels), spec)
        record = metric_record(spec.name, target, score, offsets, seed=spec.seed)
        record.update(details)
        record["label_noise_rate"] = rate
        record["training_label_flips"] = flip_count
        outputs.append(record)
    grouped = {}
    for name in ("DKT-64-Adam", "DKT-64-AdamW"):
        aucs = [row["roc_auc"] for row in outputs if row["name"] == name]
        grouped[name] = {
            "mean_roc_auc": float(np.mean(aucs)),
            "standard_deviation_roc_auc": float(np.std(aucs, ddof=1)),
            "seed_count": len(aucs),
        }
    result = {
        "experiment": "student_disjoint_training_label_noise_robustness",
        "source_sha256": "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da",
        "split": {"student_level": True, "seed": split.seed, "train_validation_test": [0.8, 0.1, 0.1]},
        "perturbation": {
            "type": "independent outcome inversion in training data only",
            "rate": rate,
            "validation_and_test_unchanged": True,
            "interpretation_limit": "This controlled perturbation is not a model of real learner behaviour or a claim about causal robustness.",
        },
        "runs": outputs,
        "aggregate": grouped,
        "privacy": "No student records, split memberships, sequence samples, or individual predictions are written to this public aggregate output.",
    }
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PRIVATE_DIR / "label_noise_robustness_private.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PUBLIC_DIR / "label_noise_robustness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
