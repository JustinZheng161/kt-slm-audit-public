"""Generate submission-ready figures from non-identifying aggregate experiment files.

All figures are derived from the public aggregate JSON outputs. No raw learner event,
student identifier, split membership, or individual prediction is read or written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if (_HERE.parents[2] / "research_code").exists() else _HERE.parents[1]
IS_PROJECT_LAYOUT = (ROOT / "research_code").exists()
RESULTS = ROOT / "research_code" / "results" if IS_PROJECT_LAYOUT else ROOT / "results"
FIGURES = ROOT / "research_artifacts" / "figures" if IS_PROJECT_LAYOUT else ROOT / "figures"
TABLES = ROOT / "research_artifacts" / "tables" if IS_PROJECT_LAYOUT else ROOT / "tables"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / name.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ablation = load("student_disjoint_baseline_summary_round2.json")
    clean = load("clean_dkt64_adam_multiseed.json")
    robustness = load("label_noise_robustness.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    TABLES.mkdir(parents=True, exist_ok=True)

    # Figure 1: interpretable baseline hierarchy with cluster-bootstrap intervals.
    core = ablation["metrics"][:3]
    labels = [record["name"] for record in core]
    aucs = [record["roc_auc"] for record in core]
    ci_low = [record["roc_auc"] - record["student_cluster_bootstrap_95_ci"][0] for record in core]
    ci_high = [record["student_cluster_bootstrap_95_ci"][1] - record["roc_auc"] for record in core]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    palette = ["#9BA7B0", "#4B7B8B", "#D0713F"]
    bars = ax.bar(labels, aucs, yerr=np.array([ci_low, ci_high]), capsize=3, color=palette, edgecolor="#1A2730", linewidth=0.6)
    ax.set_ylim(0.55, 0.82)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Student-disjoint ASSISTments2009 baseline hierarchy")
    ax.grid(axis="y", color="#DDE3E6", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, auc, upper in zip(bars, aucs, ci_high):
        ax.text(bar.get_x() + bar.get_width()/2, auc + upper + 0.006, f"{auc:.4f}", ha="center", va="bottom", fontsize=8)
    fig.text(0.01, -0.06, "80/10/10 student split; 1,000 student-cluster bootstrap replicates. DKT uses seed 20260822.", fontsize=7)
    save(fig, "fig1_student_disjoint_baselines.png")

    # Figure 2: three-seed clean ablation, capacity/optimizer separation.
    clean_aucs = [record["roc_auc"] for record in clean["runs"]]
    candidate_64 = [record["roc_auc"] for record in ablation["metrics"] if record["name"] == "DKT-64-AdamW"]
    candidate_96 = [record["roc_auc"] for record in ablation["metrics"] if record["name"] == "DKT-96-AdamW"]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    groups = [("DKT-64\nAdam", clean_aucs, "#4B7B8B"), ("DKT-64\nAdamW", candidate_64, "#83A9B5"), ("DKT-96\nAdamW", candidate_96, "#D0713F")]
    rng = np.random.default_rng(20260822)
    for index, (label, values, color) in enumerate(groups):
        jitter = rng.uniform(-0.06, 0.06, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=34, color=color, edgecolor="#1A2730", linewidth=0.5, zorder=3)
        ax.hlines(np.mean(values), index - 0.20, index + 0.20, color="#1A2730", linewidth=1.4)
        ax.text(index, 0.7602, f"mean {np.mean(values):.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.760, 0.770)
    ax.set_xticks(range(3), [group[0] for group in groups])
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Clean ablation: no observable optimization gain")
    ax.grid(axis="y", color="#DDE3E6", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.06, "Identical 80/10/10 split, seed set, selection rule, and 8-epoch cap; horizontal bars denote means.\nThe y-axis is intentionally expanded only to reveal run-to-run variation, not a practically meaningful difference.", fontsize=6.5)
    save(fig, "fig2_clean_ablation.png")

    # Figure 3: controlled training-label-flip robustness.
    clean_mean = clean["aggregate"]["mean_roc_auc"]
    robust_means = [robustness["aggregate"][name]["mean_roc_auc"] for name in ("DKT-64-Adam", "DKT-64-AdamW")]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width/2, [clean_mean, clean_mean], width, color="#4B7B8B", label="Clean training")
    ax.bar(x + width/2, robust_means, width, color="#D0713F", label="10% train-label inversion")
    for index, value in enumerate(robust_means):
        ax.text(index + width/2, value + 0.0004, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["DKT-64 Adam", "DKT-64 AdamW"])
    ax.set_ylim(0.745, 0.770)
    ax.set_ylabel("Mean test ROC-AUC across 3 seeds")
    ax.set_title("Controlled training-label perturbation robustness")
    ax.grid(axis="y", color="#DDE3E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.01, -0.06, "Validation/test data remain unchanged. This synthetic perturbation is not a model of learner behaviour.", fontsize=7)
    save(fig, "fig3_label_noise_robustness.png")

    # Figure 4: external figures arranged by source protocol to prevent invalid arithmetic gaps.
    external = [
        ("pyKT\nstandardized", "IEKT", 0.7861, "#9BA7B0"),
        ("pyKT\nstandardized", "AKT", 0.7853, "#9BA7B0"),
        ("pyKT\nstandardized", "simpleKT", 0.7744, "#9BA7B0"),
        ("AAAI UKT\nprotocol", "UKT", 0.8563, "#725C8A"),
        ("This work\ncontrolled protocol", "DKT-64 Adam", clean_mean, "#D0713F"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.25))
    labels = [f"{protocol}\n{method}" for protocol, method, _, _ in external]
    values = [value for _, _, value, _ in external]
    colors = [color for _, _, _, color in external]
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="#1A2730", linewidth=0.5)
    ax.set_ylim(0.70, 0.88)
    ax.set_ylabel("Reported ROC-AUC")
    ax.set_xticks(range(len(values)), labels, fontsize=7)
    ax.set_title("External reference values must remain protocol-separated")
    ax.grid(axis="y", color="#DDE3E6", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.004, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    fig.text(0.01, -0.08, "pyKT standardized: Liu et al. (ICLR 2023). UKT: Cheng et al. (AAAI 2025).\nThis work uses corrected collapsed CSV and a distinct 80/10/10 student split; do not subtract bars across protocols.", fontsize=6.7)
    save(fig, "fig4_protocol_separated_references.png")

    # Machine-readable summary for table generation.
    with (TABLES / "table_experiment_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "method", "test_roc_auc", "uncertainty", "protocol_note"])
        for record in ablation["metrics"][:2]:
            lower, upper = record["student_cluster_bootstrap_95_ci"]
            writer.writerow(["this_work", record["name"], f"{record['roc_auc']:.4f}", f"95% student-cluster CI [{lower:.4f}, {upper:.4f}]", "same 80/10/10 student split"])
        writer.writerow(["this_work", "DKT-64 Adam", f"{clean_mean:.4f}", f"3-seed SD {clean['aggregate']['standard_deviation_roc_auc']:.4f}", "same 80/10/10 student split"])
        writer.writerow(["this_work", "DKT-64 AdamW", f"{ablation['candidate_seed_summary']['DKT-64-AdamW']['mean_roc_auc']:.4f}", f"3-seed SD {ablation['candidate_seed_summary']['DKT-64-AdamW']['standard_deviation_roc_auc']:.4f}", "same 80/10/10 student split"])
        writer.writerow(["this_work", "DKT-96 AdamW", f"{ablation['candidate_seed_summary']['DKT-96-AdamW']['mean_roc_auc']:.4f}", f"3-seed SD {ablation['candidate_seed_summary']['DKT-96-AdamW']['standard_deviation_roc_auc']:.4f}", "same 80/10/10 student split"])


if __name__ == "__main__":
    main()
