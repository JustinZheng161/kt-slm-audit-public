"""Revision 3 figures from aggregate-only JSON outputs.

No raw learner events, identifiers, split memberships, checkpoints, or individual
predictions are read or written. All displayed values originate from aggregate JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
# Public layout: <repo>/analysis; private archive layout: <repo>/code/analysis.
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]
IS_PROJECT_LAYOUT = (ROOT / "research_code").exists()
if IS_PROJECT_LAYOUT:
    RESULTS = ROOT / "research_code" / "results"
    FIGURES = ROOT / "research_artifacts" / "figures"
    TABLES = ROOT / "research_artifacts" / "tables"
elif (ROOT / "results" / "revision3").exists():
    RESULTS = ROOT / "results" / "revision3"
    FIGURES = ROOT / "figures" / "revision3"
    TABLES = ROOT / "tables" / "revision3"
else:
    RESULTS = ROOT / "artifacts" / "revision3"
    FIGURES = ROOT / "figures" / "revision3"
    TABLES = ROOT / "tables" / "revision3"

COLORS = {"Adam": "#31708E", "AdamW": "#C76B45", "Reference": "#76818A", "BKT": "#5F8D4E"}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / name.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#56666F")
    ax.spines["bottom"].set_color("#56666F")
    ax.grid(axis="y", color="#DDE3E6", linewidth=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    main_result = load("revision3_main_eight_epoch_probability_quality.json")
    extended = load("revision3_exploratory_extended_budget.json")
    sensitivity = load("revision3_training_label_inversion_sensitivity.json")
    seed_ranges = load("revision5_three_seed_observed_ranges.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    TABLES.mkdir(parents=True, exist_ok=True)

    # Figure 1 — comparable main-analysis baselines only.
    prior = main_result["single_run_references"]["Skill-prior"]
    bkt = main_result["single_run_references"]["BKT-per-skill-EM"]
    dkt_seed_rows = main_result["clean_dkt_runs"]["DKT-64-Adam"]
    dkt_seed1 = dkt_seed_rows[0]
    records = [prior, bkt, dkt_seed1]
    labels = ["Skill-prior", "Per-skill BKT", "DKT-64\nAdam"]
    aucs = [record["roc_auc"] for record in records]
    ci_low = [record["roc_auc"] - record["student_cluster_bootstrap_95_ci"][0] for record in records]
    ci_high = [record["student_cluster_bootstrap_95_ci"][1] - record["roc_auc"] for record in records]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    bars = ax.bar(labels, aucs, yerr=np.array([ci_low, ci_high]), capsize=3, color=[COLORS["Reference"], COLORS["BKT"], COLORS["Adam"]], edgecolor="#1A2730", linewidth=0.6)
    ax.set_ylim(0.55, 0.82)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Aligned next-response baseline hierarchy")
    style_axes(ax)
    for bar, auc, upper in zip(bars, aucs, ci_high):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + upper + 0.006, f"{auc:.4f}", ha="center", va="bottom", fontsize=8)
    dkt_values = np.asarray([row["roc_auc"] for row in dkt_seed_rows])
    dkt_jitter = np.asarray([-0.075, 0.0, 0.075])
    ax.scatter(2 + dkt_jitter, dkt_values, s=22, marker="o", facecolor="white", edgecolor="#17252D", linewidth=0.7, zorder=5, label="DKT seed estimates")
    fig.text(0.01, -0.06, "24,306 shared second-and-later test interactions; fixed 80/10/10 student split; 1,000 student-cluster bootstrap replicates. The DKT bar and error bar are for seed 20260822; the three open circles show all seed-level AUC estimates (SD reported in Table 1).", fontsize=6.2)
    save(fig, "fig1_student_disjoint_baselines.png")

    # Figure 2 — three seed ablation without a truncated near-value scale.
    groups = [
        ("DKT-64\nAdam", "DKT-64-Adam", main_result["clean_dkt_runs"]["DKT-64-Adam"], COLORS["Adam"]),
        ("DKT-64\nAdamW", "DKT-64-AdamW", main_result["clean_dkt_runs"]["DKT-64-AdamW"], "#7FA7B8"),
        ("DKT-96\nAdamW + dropout", "DKT-96-AdamW-dropout", main_result["clean_dkt_runs"]["DKT-96-AdamW-dropout"], COLORS["AdamW"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    rng = np.random.default_rng(20260822)
    for index, (label, key, rows, color) in enumerate(groups):
        values = [row["roc_auc"] for row in rows]
        jitter = rng.uniform(-0.06, 0.06, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=34, color=color, edgecolor="#1A2730", linewidth=0.5, zorder=3)
        ax.hlines(np.mean(values), index - 0.20, index + 0.20, color="#1A2730", linewidth=1.4, zorder=4)
        ax.text(index, 0.751, f"mean {np.mean(values):.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.70, 0.78)
    ax.set_xticks(range(3), [group[0] for group in groups])
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Main analysis: three paired-seed ablations")
    style_axes(ax)
    fig.text(0.01, -0.06, "Identical split, seed set, validation rule, eight-epoch budget and 24,306 targets. The y-axis begins at 0.70; horizontal bars are means and points are the three observed seed estimates. This display is descriptive, not an equivalence test or population interval.", fontsize=6.1)
    save(fig, "fig2_clean_ablation.png")

    # Figure 3 — a genuine three-level sensitivity curve plus its clean reference.
    rates = np.array([0.00, 0.05, 0.10, 0.20])
    adam_summary = sensitivity["clean_reference_from_same_script"]["DKT-64-Adam"]
    adamw_summary = sensitivity["clean_reference_from_same_script"]["DKT-64-AdamW"]
    adam_values = [adam_summary["mean_roc_auc"]]
    adam_sds = [adam_summary["standard_deviation_roc_auc"]]
    adamw_values = [adamw_summary["mean_roc_auc"]]
    adamw_sds = [adamw_summary["standard_deviation_roc_auc"]]
    for rate_key in ("0.05", "0.10", "0.20"):
        adam = sensitivity["aggregate_by_rate_and_method"][rate_key]["DKT-64-Adam"]
        adamw = sensitivity["aggregate_by_rate_and_method"][rate_key]["DKT-64-AdamW"]
        adam_values.append(adam["mean_roc_auc"])
        adam_sds.append(adam["standard_deviation_roc_auc"])
        adamw_values.append(adamw["mean_roc_auc"])
        adamw_sds.append(adamw["standard_deviation_roc_auc"])
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ax.errorbar(rates, adam_values, yerr=adam_sds, marker="o", markersize=4, linewidth=1.6, capsize=3, color=COLORS["Adam"], label="DKT-64 Adam")
    ax.errorbar(rates, adamw_values, yerr=adamw_sds, marker="s", markersize=4, linewidth=1.6, linestyle="--", dashes=(3, 2), capsize=3, color=COLORS["AdamW"], label="DKT-64 AdamW")
    ax.set_xticks(rates, ["0%", "5%", "10%", "20%"])
    ax.set_ylim(0.735, 0.770)
    ax.set_ylabel("Mean test ROC-AUC across 3 seeds")
    ax.set_xlabel("Independent training-label inversion rate")
    ax.set_title("Synthetic label-inversion sensitivity")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.text(0.01, -0.07, "Error bars are across-seed SDs. Only training labels are independently inverted; validation/test labels remain clean. This is a synthetic sensitivity analysis, not a model of learner behaviour.", fontsize=6.25)
    save(fig, "fig3_label_noise_sensitivity.png")

    # Figure 4 — probability quality, separated from discrimination.
    aggregate = main_result["clean_dkt_aggregate"]
    dkt_names = ["DKT-64-Adam", "DKT-64-AdamW", "DKT-96-AdamW-dropout"]
    short_labels = ["64 Adam", "64 AdamW", "96 AdamW\n+ dropout"]
    brier = [aggregate[name]["mean_brier_score"] for name in dkt_names]
    ece = [aggregate[name]["mean_ece_10"] for name in dkt_names]
    bins = aggregate["DKT-64-Adam"]["mean_calibration_bins_10"]
    curve_x = [entry["mean_prediction_across_seeds"] for entry in bins if entry["mean_prediction_across_seeds"] is not None]
    curve_y = [entry["empirical_accuracy_across_seeds"] for entry in bins if entry["empirical_accuracy_across_seeds"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.02, 1.15]})
    ax = axes[0]
    x = np.arange(len(dkt_names))
    width = 0.36
    ax.bar(x - width / 2, brier, width, color=COLORS["Adam"], label="Brier score")
    ax.bar(x + width / 2, ece, width, color=COLORS["AdamW"], label="ECE (10 bins)")
    ax.set_xticks(x, short_labels)
    ax.set_ylim(0, 0.22)
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("Probability-quality summaries")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#879299", linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(curve_x, curve_y, marker="o", markersize=3.5, color=COLORS["Adam"], linewidth=1.5, label="DKT-64 Adam")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability curve (clean training)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    fig.text(0.01, -0.06, "All summaries use the same 24,306 test targets. ECE uses ten fixed-width probability bins; Brier score and ECE are descriptive probability-quality measures, not hypothesis tests.", fontsize=6.2)
    save(fig, "fig4_probability_quality.png")

    # Figure A1 — explicitly exploratory extended-budget diagnostic across all seeds.
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    extended_colors = ["#31708E", "#6D9EB3", "#9EC1CE"]
    for row, color in zip(extended["runs"], extended_colors):
        history = row["training_history"]
        epochs = [entry["epoch"] for entry in history]
        validation_auc = [entry["validation_auc"] for entry in history]
        seed = row["specification"]["seed"]
        selected = row["selected_epoch"]
        ax.plot(epochs, validation_auc, color=color, linewidth=1.25, marker="o", markersize=2.7, label=f"seed {seed} (select {selected})")
        ax.scatter([selected], [validation_auc[selected - 1]], marker="*", s=48, color=color, zorder=4)
    ax.axvline(8, color="#56666F", linestyle="--", linewidth=0.9, label="main 8-epoch cap")
    ax.set_xlim(1, 20)
    ax.set_ylim(0.72, 0.78)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Exploratory 20-epoch validation trajectories")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=6.0, loc="lower left")
    fig.text(0.01, -0.07, "Test data were evaluated once per seed only after selection of the maximum validation AUC within 20 epochs. This post hoc extension does not replace the pre-specified eight-epoch main analysis.", fontsize=6.15)
    save(fig, "figA1_exploratory_extended_budget.png")

    # Compact safe table for manuscript generation/inspection.
    with (TABLES / "revision3_aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["analysis", "method", "test_prediction_rows", "mean_roc_auc", "sd_roc_auc", "observed_seed_minimum", "observed_seed_maximum", "mean_brier_score", "mean_ece_10", "scope"])
        for name, values in aggregate.items():
            observed_range = seed_ranges["configurations"][name]["roc_auc"]["observed_range_min_max"]
            writer.writerow(["main_8_epoch", name, values["test_prediction_rows_per_seed"], f"{values['mean_roc_auc']:.6f}", f"{values['standard_deviation_roc_auc']:.6f}", f"{observed_range[0]:.6f}", f"{observed_range[1]:.6f}", f"{values['mean_brier_score']:.6f}", f"{values['mean_ece_10']:.6f}", "three-seed; primary budget-conditional analysis; descriptive observed min-max range"])
        writer.writerow(["exploratory_20_epoch", "DKT-64 Adam", extended["aggregate"]["test_prediction_rows_per_seed"], f"{extended['aggregate']['mean_roc_auc']:.6f}", f"{extended['aggregate']['standard_deviation_roc_auc']:.6f}", "", "", f"{extended['aggregate']['mean_brier_score']:.6f}", f"{extended['aggregate']['mean_ece_10']:.6f}", "post hoc exploratory extension"])
        for rate, methods in sensitivity["aggregate_by_rate_and_method"].items():
            for name, values in methods.items():
                writer.writerow([f"noise_{rate}", name, values["test_prediction_rows_per_seed"], f"{values['mean_roc_auc']:.6f}", f"{values['standard_deviation_roc_auc']:.6f}", "", "", f"{values['mean_brier_score']:.6f}", f"{values['mean_ece_10']:.6f}", "synthetic training-label inversion"])


if __name__ == "__main__":
    main()
