"""Run reproducible student-disjoint knowledge-tracing baselines on controlled data.

Privacy: raw events, student-level split membership, and individual predictions are
written only below private_data/. The public research_code/ output contains no raw
records or student identifiers: only source checksums, aggregate counts, settings,
and aggregate metric summaries.

The implementation deliberately distinguishes a reproducible KT benchmark from the
historical LoRA representation analysis. It does not reconstruct unavailable LoRA
checkpoints, hidden states, or next-token scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if (_HERE.parents[2] / "research_code").exists() else _HERE.parents[1]
CONTROLLED_DATA_ROOT = Path(os.environ.get("KT_AUDIT_DATA_ROOT", ROOT / "private_data"))
RAW_DATA = CONTROLLED_DATA_ROOT / "raw" / "skill_builder_data_corrected_collapsed.csv"
PRIVATE_DIR = CONTROLLED_DATA_ROOT / "results"
PUBLIC_DIR = ROOT / "research_code" / "results" if (ROOT / "research_code").exists() else ROOT / "results"


@dataclass(frozen=True)
class SplitSpec:
    seed: int = 20260822
    train_fraction: float = 0.80
    validation_fraction: float = 0.10


@dataclass(frozen=True)
class DKTSpec:
    name: str
    embedding_dim: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    seed: int


def stable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_student_raw_sequences(path: Path) -> list[list[tuple[str, int]]]:
    """Read chronological student sequences without fitting labels on held-out students."""
    required = ["user_id", "order_id", "skill_id", "correct"]
    frame = pd.read_csv(
        path,
        usecols=required,
        dtype={"user_id": "string", "order_id": "Int64", "skill_id": "string", "correct": "Int8"},
        encoding="utf-8",
        encoding_errors="replace",
    )
    frame = frame.dropna(subset=required)
    frame = frame[frame["correct"].isin([0, 1])].copy()
    frame["skill_id"] = frame["skill_id"].str.strip()
    frame = frame[frame["skill_id"].ne("")]
    frame = frame.sort_values(["user_id", "order_id"], kind="mergesort")
    sequences: list[list[tuple[str, int]]] = []
    for _, group in frame.groupby("user_id", sort=False):
        sequence = [(str(skill), int(correct)) for skill, correct in zip(group["skill_id"], group["correct"])]
        if len(sequence) >= 2:
            sequences.append(sequence)
    if not sequences:
        raise ValueError("No valid student sequences with at least two interactions were constructed.")
    return sequences


def encode_sequences(sequences: list[list[tuple[str, int]]], skill_index: dict[str, int]) -> list[list[tuple[int, int]]]:
    unseen = {skill for sequence in sequences for skill, _ in sequence if skill not in skill_index}
    if unseen:
        raise ValueError(f"Validation/test include {len(unseen)} labels not present in the train-fitted vocabulary.")
    return [[(skill_index[skill], outcome) for skill, outcome in sequence] for sequence in sequences]


def load_train_fitted_splits(
    path: Path, spec: SplitSpec
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[str], dict]:
    """Split raw students first, then fit the categorical skill vocabulary on training students only."""
    raw_sequences = load_student_raw_sequences(path)
    raw_train, raw_validation, raw_test = split_students(raw_sequences, spec)
    skill_labels = sorted({skill for sequence in raw_train for skill, _ in sequence})
    skill_index = {label: index for index, label in enumerate(skill_labels)}
    train_labels = set(skill_index)
    validation_labels = {skill for sequence in raw_validation for skill, _ in sequence}
    test_labels = {skill for sequence in raw_test for skill, _ in sequence}
    validation_unseen = validation_labels.difference(train_labels)
    test_unseen = test_labels.difference(train_labels)
    if validation_unseen or test_unseen:
        raise ValueError(
            "The fixed student-disjoint split is unsupported by the train-fitted vocabulary: "
            f"validation unseen={len(validation_unseen)}, test unseen={len(test_unseen)}."
        )
    support = {
        "vocabulary_fit": "The atomic skill_id vocabulary is fitted on training students only after student-level splitting.",
        "train_unique_skill_labels": len(train_labels),
        "validation_unique_skill_labels": len(validation_labels),
        "test_unique_skill_labels": len(test_labels),
        "validation_labels_not_seen_in_train": len(validation_unseen),
        "test_labels_not_seen_in_train": len(test_unseen),
        "validation_label_coverage_by_train": len(validation_labels.intersection(train_labels)) / len(validation_labels) if validation_labels else None,
        "test_label_coverage_by_train": len(test_labels.intersection(train_labels)) / len(test_labels) if test_labels else None,
    }
    return (
        encode_sequences(raw_train, skill_index),
        encode_sequences(raw_validation, skill_index),
        encode_sequences(raw_test, skill_index),
        skill_labels,
        support,
    )


def load_student_sequences(path: Path) -> tuple[list[list[tuple[int, int]]], list[str]]:
    """Legacy whole-file encoder retained for external inspection; not used in controlled experiments."""
    raw_sequences = load_student_raw_sequences(path)
    skill_labels = sorted({skill for sequence in raw_sequences for skill, _ in sequence})
    skill_index = {label: index for index, label in enumerate(skill_labels)}
    return encode_sequences(raw_sequences, skill_index), skill_labels


def split_students(sequences: list[list[tuple[int, int]]], spec: SplitSpec) -> tuple[list, list, list]:
    order = np.random.default_rng(spec.seed).permutation(len(sequences))
    train_cut = int(len(order) * spec.train_fraction)
    validation_cut = int(len(order) * (spec.train_fraction + spec.validation_fraction))
    return (
        [sequences[index] for index in order[:train_cut]],
        [sequences[index] for index in order[train_cut:validation_cut]],
        [sequences[index] for index in order[validation_cut:]],
    )


def auc_or_nan(target: np.ndarray, score: np.ndarray) -> float:
    if len(target) == 0 or len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, score))


def student_bootstrap_ci(
    targets: np.ndarray,
    scores: np.ndarray,
    student_offsets: list[tuple[int, int]],
    seed: int,
    replicates: int = 1000,
) -> tuple[float, float]:
    """Cluster bootstrap by student, preserving within-student serial dependence."""
    if not student_offsets:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    n = len(student_offsets)
    for replicate in range(replicates):
        sampled = rng.integers(0, n, size=n)
        indices = np.concatenate([np.arange(student_offsets[i][0], student_offsets[i][1]) for i in sampled])
        values[replicate] = auc_or_nan(targets[indices], scores[indices])
    return (float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975)))


def make_skill_priors(train_sequences: list[list[tuple[int, int]]], skill_count: int, alpha: float = 1.0) -> np.ndarray:
    correct = np.zeros(skill_count, dtype=float)
    count = np.zeros(skill_count, dtype=float)
    for sequence in train_sequences:
        for skill, outcome in sequence:
            correct[skill] += outcome
            count[skill] += 1
    return (correct + alpha) / (count + 2.0 * alpha)


def evaluate_skill_prior(test_sequences: list[list[tuple[int, int]]], priors: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    targets: list[int] = []
    scores: list[float] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sequence in test_sequences:
        # Score only second and later events so all methods share the next-response target set.
        for skill, outcome in sequence[1:]:
            targets.append(outcome)
            scores.append(float(priors[skill]))
        offsets.append((cursor, len(targets)))
        cursor = len(targets)
    return np.asarray(targets), np.asarray(scores), offsets


def per_skill_response_sequences(sequences: list[list[tuple[int, int]]]) -> dict[int, list[np.ndarray]]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for sequence in sequences:
        by_skill: dict[int, list[int]] = defaultdict(list)
        for skill, outcome in sequence:
            by_skill[skill].append(outcome)
        for skill, outcomes in by_skill.items():
            grouped[skill].append(np.asarray(outcomes, dtype=np.int8))
    return grouped


def bkt_em(sequences: Iterable[np.ndarray], iterations: int = 20) -> np.ndarray:
    """Fit 2-state BKT via EM with bounded parameters [P(L0), P(T), P(G), P(S)]."""
    seqs = [sequence for sequence in sequences if len(sequence) > 0]
    if sum(len(sequence) for sequence in seqs) < 20:
        return np.array([0.25, 0.10, 0.20, 0.10], dtype=float)
    params = np.array([0.25, 0.12, 0.20, 0.10], dtype=float)
    eps = 1e-8
    for _ in range(iterations):
        initial_mastery = transition_mastery = transition_unmastered = 0.0
        unmastered_observed_correct = unmastered_weight = 0.0
        mastered_observed_incorrect = mastered_weight = 0.0
        pi, learn, guess, slip = params
        transition = np.array([[1.0 - learn, learn], [0.0, 1.0]], dtype=float)
        emission = np.array([[1.0 - guess, guess], [slip, 1.0 - slip]], dtype=float)
        for observed in seqs:
            length = len(observed)
            forward = np.zeros((length, 2), dtype=float)
            backward = np.zeros((length, 2), dtype=float)
            forward[0] = np.array([1.0 - pi, pi]) * emission[:, observed[0]]
            forward[0] /= forward[0].sum() + eps
            for step in range(1, length):
                forward[step] = (forward[step - 1] @ transition) * emission[:, observed[step]]
                forward[step] /= forward[step].sum() + eps
            backward[-1] = 1.0
            for step in range(length - 2, -1, -1):
                backward[step] = transition @ (emission[:, observed[step + 1]] * backward[step + 1])
                backward[step] /= backward[step].sum() + eps
            gamma = forward * backward
            gamma /= gamma.sum(axis=1, keepdims=True) + eps
            initial_mastery += gamma[0, 1]
            unmastered_weight += gamma[:, 0].sum()
            mastered_weight += gamma[:, 1].sum()
            unmastered_observed_correct += gamma[observed == 1, 0].sum()
            mastered_observed_incorrect += gamma[observed == 0, 1].sum()
            for step in range(length - 1):
                xi = np.outer(forward[step], emission[:, observed[step + 1]] * backward[step + 1]) * transition
                xi /= xi.sum() + eps
                transition_unmastered += xi[0].sum()
                transition_mastery += xi[0, 1]
        params = np.array(
            [
                np.clip(initial_mastery / len(seqs), 0.01, 0.99),
                np.clip(transition_mastery / (transition_unmastered + eps), 0.001, 0.50),
                np.clip(unmastered_observed_correct / (unmastered_weight + eps), 0.001, 0.49),
                np.clip(mastered_observed_incorrect / (mastered_weight + eps), 0.001, 0.49),
            ]
        )
    return params


def fit_bkt_models(train_sequences: list[list[tuple[int, int]]], skill_count: int) -> np.ndarray:
    by_skill = per_skill_response_sequences(train_sequences)
    return np.stack([bkt_em(by_skill.get(skill, [])) for skill in range(skill_count)])


def bkt_predictions(test_sequences: list[list[tuple[int, int]]], parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    targets: list[int] = []
    scores: list[float] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sequence in test_sequences:
        mastery = parameters[:, 0].copy()
        for event_index, (skill, outcome) in enumerate(sequence):
            _, learn, guess, slip = parameters[skill]
            probability = mastery[skill] * (1.0 - slip) + (1.0 - mastery[skill]) * guess
            if event_index > 0:
                targets.append(outcome)
                scores.append(float(probability))
            posterior = (mastery[skill] * ((1.0 - slip) if outcome else slip)) / (
                probability if outcome else (1.0 - probability)
            )
            mastery[skill] = posterior + (1.0 - posterior) * learn
        offsets.append((cursor, len(targets)))
        cursor = len(targets)
    return np.asarray(targets), np.asarray(scores), offsets


class NextInteractionDataset(Dataset):
    """Chunks each student sequence but never mixes student memberships across splits."""

    def __init__(self, sequences: list[list[tuple[int, int]]], skill_count: int, max_length: int = 200) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for sequence in sequences:
            for start in range(0, len(sequence) - 1, max_length):
                chunk = sequence[start : start + max_length + 1]
                if len(chunk) < 2:
                    continue
                skills = np.asarray([event[0] for event in chunk], dtype=np.int64)
                outcomes = np.asarray([event[1] for event in chunk], dtype=np.int64)
                tokens = skills[:-1] + outcomes[:-1] * skill_count
                self.samples.append((tokens, skills[1:], outcomes[1:]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.samples[index]


def collate_batch(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[Tensor, Tensor, Tensor]:
    length = max(len(tokens) for tokens, _, _ in batch)
    tokens = torch.zeros((len(batch), length), dtype=torch.long)
    target_skills = torch.full((len(batch), length), -1, dtype=torch.long)
    target_outcomes = torch.zeros((len(batch), length), dtype=torch.float32)
    for row, (input_tokens, skills, outcomes) in enumerate(batch):
        end = len(input_tokens)
        tokens[row, :end] = torch.from_numpy(input_tokens)
        target_skills[row, :end] = torch.from_numpy(skills)
        target_outcomes[row, :end] = torch.from_numpy(outcomes.astype(np.float32))
    return tokens, target_skills, target_outcomes


class DKTModel(nn.Module):
    """A compact DKT baseline with interaction tokens and next-skill output logits."""

    def __init__(self, skill_count: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(skill_count * 2, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, skill_count)

    def forward(self, tokens: Tensor) -> Tensor:
        states, _ = self.gru(self.embedding(tokens))
        return self.output(self.dropout(states))


def batch_loss(model: DKTModel, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    tokens, target_skills, target_outcomes = batch
    logits = model(tokens)
    mask = target_skills.ge(0)
    selected = logits.gather(2, target_skills.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(selected[mask], target_outcomes[mask])
    return loss, selected[mask].detach(), target_outcomes[mask].detach()


def evaluate_dkt(model: DKTModel, sequences: list[list[tuple[int, int]]], skill_count: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    model.eval()
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    with torch.no_grad():
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            skills = torch.tensor([event[0] for event in sequence], dtype=torch.long)
            outcomes = torch.tensor([event[1] for event in sequence], dtype=torch.long)
            tokens = (skills[:-1] + outcomes[:-1] * skill_count).unsqueeze(0)
            logits = model(tokens).squeeze(0)
            selected = logits.gather(1, skills[1:].unsqueeze(1)).squeeze(1)
            local_scores = torch.sigmoid(selected).cpu().numpy()
            local_targets = outcomes[1:].cpu().numpy()
            targets.append(local_targets)
            scores.append(local_scores)
            offsets.append((cursor, cursor + len(local_targets)))
            cursor += len(local_targets)
    return np.concatenate(targets), np.concatenate(scores), offsets


def train_dkt(
    train_sequences: list[list[tuple[int, int]]],
    validation_sequences: list[list[tuple[int, int]]],
    test_sequences: list[list[tuple[int, int]]] | None,
    skill_count: int,
    spec: DKTSpec,
    record_training_auc: bool = False,
    early_stop_patience: int | None = 3,
) -> tuple[dict, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    set_seed(spec.seed)
    model = DKTModel(skill_count, spec.embedding_dim, spec.hidden_dim, spec.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    loader = DataLoader(NextInteractionDataset(train_sequences, skill_count), batch_size=64, shuffle=True, collate_fn=collate_batch)
    best_state: dict[str, Tensor] | None = None
    best_validation = -math.inf
    patience = 0
    history: list[dict] = []
    for epoch in range(1, spec.epochs + 1):
        model.train()
        total_loss = 0.0
        observed = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, labels = batch_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            observed += len(labels)
        val_targets, val_scores, _ = evaluate_dkt(model, validation_sequences, skill_count)
        validation_auc = auc_or_nan(val_targets, val_scores)
        record = {"epoch": epoch, "training_bce": total_loss / max(observed, 1), "validation_auc": validation_auc}
        if record_training_auc:
            train_targets, train_scores, _ = evaluate_dkt(model, train_sequences, skill_count)
            record["training_auc"] = auc_or_nan(train_targets, train_scores)
        history.append(record)
        if validation_auc > best_validation + 1e-6:
            best_validation = validation_auc
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if early_stop_patience is not None and patience >= early_stop_patience:
                break
    if best_state is None:
        raise RuntimeError("DKT did not produce a valid validation checkpoint.")
    model.load_state_dict(best_state)
    if test_sequences is None:
        targets, scores, offsets = np.asarray([], dtype=np.int64), np.asarray([], dtype=float), []
    else:
        targets, scores, offsets = evaluate_dkt(model, test_sequences, skill_count)
    return (
        {
            "specification": asdict(spec),
            "selected_validation_auc": best_validation,
            "training_history": history,
            "selected_epoch": int(max(history, key=lambda item: item["validation_auc"])["epoch"]),
        },
        targets,
        scores,
        offsets,
    )


def metric_record(name: str, targets: np.ndarray, scores: np.ndarray, offsets: list[tuple[int, int]], seed: int) -> dict:
    lower, upper = student_bootstrap_ci(targets, scores, offsets, seed=seed)
    return {
        "name": name,
        "test_prediction_rows": int(len(targets)),
        "roc_auc": auc_or_nan(targets, scores),
        "student_cluster_bootstrap_95_ci": [lower, upper],
        "bootstrap_replicates": 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkt-epochs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Controlled source data missing: {RAW_DATA}")
    torch.set_num_threads(max(1, args.threads))
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    split_spec = SplitSpec()
    train_sequences, validation_sequences, test_sequences, labels, label_support = load_train_fitted_splits(RAW_DATA, split_spec)
    priors = make_skill_priors(train_sequences, len(labels))
    prior_y, prior_s, prior_offsets = evaluate_skill_prior(test_sequences, priors)
    bkt_parameters = fit_bkt_models(train_sequences, len(labels))
    bkt_y, bkt_s, bkt_offsets = bkt_predictions(test_sequences, bkt_parameters)

    baseline_spec = DKTSpec("DKT-64-Adam", 64, 64, 0.0, 0.002, 0.0, args.dkt_epochs, 20260822)
    optimized_specs = [
        *[
            DKTSpec("DKT-64-AdamW", 64, 64, 0.0, 0.002, 0.0001, args.dkt_epochs, seed)
            for seed in (20260822, 20260823, 20260824)
        ],
        *[
            DKTSpec("DKT-96-AdamW", 96, 96, 0.10, 0.001, 0.0001, args.dkt_epochs, seed)
            for seed in (20260822, 20260823, 20260824)
        ],
    ]
    dkt_runs: list[dict] = []
    private_scores: dict[str, dict[str, np.ndarray]] = {
        "skill_prior": {"target": prior_y, "score": prior_s},
        "bkt": {"target": bkt_y, "score": bkt_s},
    }
    dkt_details, dkt_y, dkt_s, dkt_offsets = train_dkt(
        train_sequences, validation_sequences, test_sequences, len(labels), baseline_spec
    )
    dkt_runs.append({**metric_record(baseline_spec.name, dkt_y, dkt_s, dkt_offsets, baseline_spec.seed), **dkt_details})
    private_scores[baseline_spec.name] = {"target": dkt_y, "score": dkt_s}
    for spec in optimized_specs:
        details, y, score, offsets = train_dkt(train_sequences, validation_sequences, test_sequences, len(labels), spec)
        dkt_runs.append({**metric_record(spec.name, y, score, offsets, spec.seed), **details})
        private_scores[f"{spec.name}_seed_{spec.seed}"] = {"target": y, "score": score}

    baseline_metrics = [
        metric_record("Skill-prior", prior_y, prior_s, prior_offsets, split_spec.seed),
        metric_record("BKT (per-skill EM)", bkt_y, bkt_s, bkt_offsets, split_spec.seed + 1),
        *dkt_runs,
    ]
    candidate_seed_summary = {}
    for candidate_name in ("DKT-64-AdamW", "DKT-96-AdamW"):
        candidate_aucs = [row["roc_auc"] for row in dkt_runs if row["name"] == candidate_name]
        candidate_seed_summary[candidate_name] = {
            "mean_roc_auc": float(np.mean(candidate_aucs)),
            "standard_deviation_roc_auc": float(np.std(candidate_aucs, ddof=1)) if len(candidate_aucs) > 1 else 0.0,
            "seed_count": len(candidate_aucs),
        }
    result = {
        "experiment": "student_disjoint_assistments2009_kt_baselines",
        "data": {
            "source_file": RAW_DATA.name,
            "source_sha256": stable_sha256(RAW_DATA),
            "source_page": "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
            "encoding": "corrected collapsed skill_id labels treated as categorical labels; multi-skill labels are not decomposed",
            "student_count": sum(split_spec_counts for split_spec_counts in [len(train_sequences), len(validation_sequences), len(test_sequences)]),
            "skill_label_count": len(labels),
            "split_student_counts": {"train": len(train_sequences), "validation": len(validation_sequences), "test": len(test_sequences)},
            "train_fitted_label_support": label_support,
        },
        "split": asdict(split_spec),
        "methods": {
            "Skill-prior": "Laplace-smoothed per-skill correctness mean fitted on training students; scored only on second and later test interactions.",
            "BKT (per-skill EM)": "Two-state BKT independently fitted per skill on training students; first test interaction updates state but is not scored.",
            "DKT-64-Adam": "GRU next-interaction baseline selected by validation AUC; scores only second and later test interactions.",
            "DKT-64-AdamW": "Decoupled-weight-decay ablation with the same capacity and learning rate as DKT-64-Adam; three independent seeds.",
            "DKT-96-AdamW": "Capacity/dropout/AdamW candidate, three independent initialisation seeds; this is a within-protocol candidate rather than an external SOTA claim.",
        },
        "metrics": baseline_metrics,
        "candidate_seed_summary": candidate_seed_summary,
        "privacy": "Raw data, student identities, split membership, per-student sequences, and row-level predictions are stored only in the controlled data root and excluded from Git.",
    }
    np.savez_compressed(PRIVATE_DIR / "student_level_predictions.npz", **{f"{name}_{key}": value for name, values in private_scores.items() for key, value in values.items()})
    (PRIVATE_DIR / "bkt_parameters_private.json").write_text(json.dumps(bkt_parameters.tolist()), encoding="utf-8")
    (PRIVATE_DIR / "experiment_summary_private.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PUBLIC_DIR / "student_disjoint_baseline_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
