#!/usr/bin/env python3
"""Reconstruct all KT paper figures in a compact Nature-style visual system.

Only aggregate-safe JSON summaries are read. The script neither reads nor writes raw
student records, identifiers, split membership, student-level predictions, per-skill
parameter vectors, bootstrap replicates, or model checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]
RESULTS = ROOT / "results" / "revision3" if (ROOT / "results" / "revision3").exists() else ROOT / "artifacts" / "revision3"
DATA = ROOT / "data" / "figure-data-kt-nature-v31.json" if (ROOT / "data").exists() else ROOT / "artifacts" / "figure_data" / "figure-data-kt-nature-v31.json"
OUT = ROOT / "figures" / "nature-kt-v31"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1B4F72"
TEAL = "#2A7F7F"
ORANGE = "#C65D3B"
GREY = "#6E7781"
INK = "#20252B"
GRID = "#D9DEE3"
PALE_BLUE = "#DCEAF2"
PALETTE = [GREY, TEAL, NAVY, ORANGE]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.4,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def first_existing(*names: str) -> Path:
    for name in names:
        candidate = RESULTS / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No expected aggregate result exists: {names}")


def load(*names: str) -> tuple[dict, Path]:
    path = first_existing(*names)
    return json.loads(path.read_text(encoding="utf-8")), path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.12, 1.06, text, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top")


def finish(fig: plt.Figure, stem: str, source_files: list[Path], description: str, uncertainty: str) -> dict:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    svg = OUT / f"{stem}.svg"
    tif = OUT / f"{stem}.tif"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    # Matplotlib may wrap SVG path data with trailing spaces; trim them so strict Git whitespace checks pass without changing geometry.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-r", "600", "-tiff", "-singlefile", str(pdf), str(tif.with_suffix(""))], check=True)
    with Image.open(tif) as image:
        dimensions = list(image.size)
        dpi = [round(float(v)) for v in image.info.get("dpi", (600, 600))]
    return {
        "file_prefix": stem,
        "display_name": stem.replace("figure-", "Figure ").replace("-", " "),
        "aggregate_inputs": [str(path.relative_to(ROOT)) for path in source_files],
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "outputs": {key: str((OUT / f"{stem}.{key}").relative_to(ROOT)) for key in ("pdf", "svg", "png", "tif")},
        "tiff_pixel_dimensions": dimensions,
        "tiff_dpi": dpi,
        "description": description,
        "uncertainty_definition": uncertainty,
        "privacy_scope": "Aggregate-only. No raw records, student identifiers, memberships, sequences, predictions, per-skill parameter vectors, bootstrap replicates or checkpoints.",
    }


def collect_data() -> tuple[dict, list[Path]]:
    main, main_path = load("revision3-main-eight-epoch-probability-quality.json", "revision3_main_eight_epoch_probability_quality.json")
    window, window_path = load("revision5-training-window-length-sensitivity.json", "revision5_training_window_length_sensitivity.json")
    noise, noise_path = load("revision3-training-label-inversion-sensitivity.json", "revision3_training_label_inversion_sensitivity.json")
    extended, extended_path = load("revision3-exploratory-extended-budget.json", "revision3_exploratory_extended_budget.json")
    audit, audit_path = load("revision6-statistical-and-bkt-audits.json", "revision6_statistical_and_bkt_audits.json")
    payload = {"main": main, "window": window, "noise": noise, "extended": extended, "audit": audit}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps({"revision": "nature-kt-v31", "privacy": "aggregate-only", "data": payload}, indent=2) + "\n", encoding="utf-8")
    return payload, [main_path, window_path, noise_path, extended_path, audit_path, DATA]


def fig1(data: dict, sources: list[Path]) -> dict:
    main = data["main"]
    records = [main["single_run_references"]["Skill-prior"], main["single_run_references"]["BKT-per-skill-EM"], main["clean_dkt_runs"]["DKT-64-Adam"][0]]
    labels = ["Skill prior", "Per-skill BKT", "DKT-64\nAdam"]
    auc = np.array([entry["roc_auc"] for entry in records])
    lo = auc - np.array([entry["student_cluster_bootstrap_95_ci"][0] for entry in records])
    hi = np.array([entry["student_cluster_bootstrap_95_ci"][1] for entry in records]) - auc
    fig, ax = plt.subplots(figsize=(6.75, 2.85))
    x = np.arange(3)
    bars = ax.bar(x, auc, yerr=np.vstack([lo, hi]), capsize=3, color=PALETTE[:3], width=0.68, edgecolor="white", linewidth=0.5, zorder=2)
    seed_auc = np.array([row["roc_auc"] for row in main["clean_dkt_runs"]["DKT-64-Adam"]])
    ax.scatter(2 + np.array([-0.085, 0.0, 0.085]), seed_auc, s=24, facecolor="white", edgecolor=INK, linewidth=0.75, zorder=4)
    for bar, value, upper in zip(bars, auc, hi):
        ax.text(bar.get_x() + bar.get_width() / 2, value + upper + 0.006, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.55, 0.82)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Student-disjoint next-response baselines", loc="left", pad=8)
    clean_axes(ax)
    ax.legend(handles=[
        Patch(facecolor=NAVY, edgecolor="white", label="Bar: single-run ROC-AUC"),
        Line2D([0], [0], color=INK, marker="|", markersize=11, linewidth=1, label="Error bar: student-cluster 95% interval"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="white", markeredgecolor=INK, markersize=5, label="Open circles: all DKT seeds"),
    ], frameon=False, loc="upper left", handlelength=1.5)
    fig.text(0.01, 0.01, "24,306 shared second-and-later test interactions; DKT bar/error bar: seed 20260822; circles: all three DKT seed estimates.", fontsize=6.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-01-student-disjoint-baselines", sources[:1], "Shared-target baseline hierarchy.", "Bars use 1,000 student-cluster bootstrap 95% intervals; open circles are the literal three DKT seed estimates.")


def fig2(data: dict, sources: list[Path]) -> dict:
    aggregate = data["window"]["aggregate"]
    keys = ["window_200", "window_500", "full_available_history"]
    labels = ["200", "500", "Full\n(1,028)"]
    means = np.array([aggregate[key]["mean_roc_auc"] for key in keys])
    sds = np.array([aggregate[key]["standard_deviation_roc_auc"] for key in keys])
    fig, ax = plt.subplots(figsize=(5.3, 2.85))
    x = np.arange(3)
    ax.errorbar(x, means, yerr=sds, color=NAVY, marker="o", markersize=5, linewidth=1.3, capsize=3, zorder=3)
    for pos, value in zip(x, means): ax.text(pos, value + 0.0012, f"{value:.4f}", ha="center", fontsize=6.9)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Maximum training window (transitions)")
    ax.set_ylabel("Mean test ROC-AUC across 3 seeds")
    ax.set_ylim(0.75, 0.78)
    ax.set_title("Training-window sensitivity", loc="left", pad=8)
    clean_axes(ax)
    fig.text(0.01, 0.01, "Capped error bars are across-seed SDs. Full-history evaluation is fixed for all conditions; this is descriptive, not an equivalence test.", fontsize=6.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-02-training-window-sensitivity", [sources[1]], "Fixed-protocol training-window sensitivity.", "Capped error bars are across-seed standard deviations (n=3), not confidence intervals.")


def fig3(data: dict, sources: list[Path]) -> dict:
    runs = data["main"]["clean_dkt_runs"]
    groups = [("DKT-64\nAdam", "DKT-64-Adam", NAVY), ("DKT-64\nAdamW", "DKT-64-AdamW", TEAL), ("DKT-96\nAdamW + dropout", "DKT-96-AdamW-dropout", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.0, 2.85))
    rng = np.random.default_rng(20260822)
    for index, (label, key, color) in enumerate(groups):
        values = np.array([row["roc_auc"] for row in runs[key]])
        ax.scatter(np.full(values.size, index) + rng.uniform(-0.075, 0.075, values.size), values, s=27, facecolor=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="_", color=INK, markersize=15, capsize=3, linewidth=0.9, zorder=4)
        ax.text(index, values.mean() - 0.006, f"mean {values.mean():.4f}", ha="center", fontsize=6.7)
    ax.set_xticks(np.arange(3), [item[0] for item in groups])
    ax.set_ylim(0.70, 0.78)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Primary ablation: paired seed observations", loc="left", pad=8)
    clean_axes(ax)
    fig.text(0.01, 0.01, "Points are all three seed-level AUCs; horizontal underscore markers show means and capped vertical error bars show across-seed SDs. Descriptive comparison only.", fontsize=6.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-03-primary-ablation", sources[:1], "Primary DKT ablation observations.", "Points are three seed estimates; horizontal underscore markers show configuration means; capped vertical error bars are across-seed SDs, not confidence intervals.")


def fig4(data: dict, sources: list[Path]) -> dict:
    noise = data["noise"]
    rates = [0.00, 0.05, 0.10, 0.20]
    labels = ["0%", "5%", "10%", "20%"]
    fig, ax = plt.subplots(figsize=(5.5, 2.85))
    for method, color, marker, line in [("DKT-64-Adam", NAVY, "o", "-"), ("DKT-64-AdamW", ORANGE, "s", "--")]:
        clean = noise["clean_reference_from_same_script"][method]
        values = [clean["mean_roc_auc"]]
        sds = [clean["standard_deviation_roc_auc"]]
        for rate in ("0.05", "0.10", "0.20"):
            entry = noise["aggregate_by_rate_and_method"][rate][method]
            values.append(entry["mean_roc_auc"]); sds.append(entry["standard_deviation_roc_auc"])
        ax.errorbar(rates, values, yerr=sds, color=color, marker=marker, markersize=4.5, linewidth=1.35, capsize=3, linestyle=line, label=method.replace("DKT-", ""))
    ax.set_xticks(rates, labels)
    ax.set_xlabel("Independent training-label inversion rate")
    ax.set_ylabel("Mean test ROC-AUC across 3 seeds")
    ax.set_ylim(0.735, 0.770)
    ax.set_title("Synthetic label-inversion sensitivity", loc="left", pad=8)
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.01, 0.01, "Capped error bars are across-seed SDs. Validation and test labels remain clean; this is a synthetic sensitivity analysis.", fontsize=6.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-04-label-inversion-sensitivity", [sources[2]], "Synthetic training-label inversion sensitivity.", "Capped error bars are across-seed standard deviations (n=3).")


def figs1(data: dict, sources: list[Path]) -> dict:
    aggregate = data["main"]["clean_dkt_aggregate"]
    configs = ["DKT-64-Adam", "DKT-64-AdamW", "DKT-96-AdamW-dropout"]
    labels = ["64 Adam", "64 AdamW", "96 AdamW\n+ dropout"]
    brier = [aggregate[key]["mean_brier_score"] for key in configs]
    ece = [aggregate[key]["mean_ece_10"] for key in configs]
    bins = aggregate[configs[0]]["mean_calibration_bins_10"]
    xs = [row["mean_prediction_across_seeds"] for row in bins if row["mean_prediction_across_seeds"] is not None]
    ys = [row["empirical_accuracy_across_seeds"] for row in bins if row["empirical_accuracy_across_seeds"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.65), gridspec_kw={"width_ratios": [1.02, 1.1]})
    x = np.arange(3); width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, brier, width, color=NAVY, label="Brier score")
    ax.bar(x + width/2, ece, width, color=ORANGE, label="ECE10")
    ax.set_xticks(x, labels); ax.set_ylabel("Score (lower is better)"); ax.set_ylim(0, 0.22); ax.set_title("Probability-quality summaries", loc="left", pad=8); clean_axes(ax); ax.legend(frameon=False, loc="upper left"); panel_label(ax, "a")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=(0, (3, 2)), linewidth=0.9, label="Perfect calibration")
    ax.plot(xs, ys, color=TEAL, marker="o", markersize=3.7, linewidth=1.3, label="DKT-64 Adam")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Empirical accuracy")
    ax.set_title("Reliability curve", loc="left", pad=8); clean_axes(ax); ax.legend(frameon=False, loc="upper left"); panel_label(ax, "b")
    fig.text(0.01, 0.01, "All summaries use the same 24,306 targets. ECE10 uses ten fixed-width bins; both measures are descriptive probability-quality summaries.", fontsize=6.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-s01-probability-quality", sources[:1], "Probability-quality summaries and reliability curve.", "No inferential interval is depicted; calibration values are descriptive aggregate summaries.")


def figs2(data: dict, sources: list[Path]) -> dict:
    fig, ax = plt.subplots(figsize=(5.8, 2.85))
    colors = [NAVY, TEAL, "#76A5AF"]
    for row, color in zip(data["extended"]["runs"], colors):
        history = row["training_history"]
        epochs = [item["epoch"] for item in history]
        values = [item["validation_auc"] for item in history]
        selected = row["selected_epoch"]
        ax.plot(epochs, values, color=color, marker="o", markersize=2.6, linewidth=1.15, label=f"Seed {row['specification']['seed']} (select {selected})")
        ax.scatter([selected], [values[selected-1]], marker="*", s=35, color=color, zorder=4)
    ax.axvline(8, color=GREY, linestyle=(0, (3, 2)), linewidth=0.9, label="Primary 8-epoch cap")
    ax.set(xlim=(1, 20), ylim=(0.72, 0.78), xlabel="Epoch", ylabel="Validation ROC-AUC")
    ax.set_title("Exploratory 20-epoch validation trajectories", loc="left", pad=8)
    clean_axes(ax); ax.legend(frameon=False, fontsize=5.9, loc="lower left")
    fig.text(0.01, 0.01, "Stars indicate the selected validation checkpoint. The post hoc extension does not replace the pre-specified eight-epoch primary analysis.", fontsize=6.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-s02-exploratory-budget", [sources[3]], "Exploratory validation trajectories under extended budget.", "Each line is one seed; no test-series error bar is used because selection precedes one test evaluation per seed.")


def figs3(data: dict, sources: list[Path]) -> dict:
    bkt = data["audit"]["bkt_diagnostics"]
    iteration = data["audit"]["bkt_iteration_sensitivity"]
    names = ["P(L0)", "P(T)", "P(G)", "P(S)"]
    quantiles = bkt["parameter_quantiles_across_skills"]
    med = np.array([quantiles[name]["p50"] for name in names])
    low = med - np.array([quantiles[name]["p05"] for name in names])
    high = np.array([quantiles[name]["p95"] for name in names]) - med
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.65), gridspec_kw={"width_ratios": [1.15, 0.9]})
    ax = axes[0]
    x = np.arange(4)
    ax.errorbar(x, med, yerr=np.vstack([low, high]), fmt="o", color=NAVY, capsize=3, markersize=4, linewidth=1.0)
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.set_ylabel("Across-skill parameter summary"); ax.set_title("BKT parameter quantiles", loc="left", pad=8); clean_axes(ax); panel_label(ax, "a")
    ax = axes[1]
    rates = [100*bkt["final_iteration_max_absolute_parameter_change"]["proportion_at_or_below_threshold"], 100*iteration["fixed_100_final_iteration_max_absolute_parameter_change"]["proportion_at_or_below_threshold"]]
    bars = ax.bar([0,1], rates, color=[GREY, TEAL], width=0.6)
    ax.set_xticks([0,1], ["20", "100"]); ax.set_xlabel("Fixed EM iterations"); ax.set_ylabel("Skills with final update ≤10⁻⁵ (%)"); ax.set_ylim(0, 100); ax.set_title("Fixed-iteration stability proxy", loc="left", pad=8); clean_axes(ax); panel_label(ax, "b")
    for bar, value in zip(bars, rates): ax.text(bar.get_x()+bar.get_width()/2, value+3, f"{value:.1f}%", ha="center", fontsize=6.9)
    fig.text(0.01, 0.01, "a, P05–P95 intervals across 149 skills after 20 fixed EM iterations. b, final-update stability proxy; not a likelihood-convergence certificate. No skill identifiers are shown.", fontsize=6.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return finish(fig, "figure-s03-bkt-diagnostics", [sources[4]], "Aggregate BKT parameter and stability diagnostics.", "P05–P95 spans are cross-skill aggregate summaries; bars are aggregate counts/proportions, not learner-level estimates.")


def main() -> None:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required for print TIFF export")
    data, sources = collect_data()
    figures = [fig1(data, sources), fig2(data, sources), fig3(data, sources), fig4(data, sources), figs1(data, sources), figs2(data, sources), figs3(data, sources)]
    manifest = {"revision": "nature-kt-v31", "style": "compact Nature-style scientific figures", "figure_count": len(figures), "figures": figures}
    (OUT / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "figure_count": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
