"""Render aggregate-only Figure 4 for the Revision 5 window-length sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]


def result_directory(root: Path) -> Path:
    candidates = (root / "research_code" / "results", root / "results" / "revision3", root / "artifacts" / "revision3")
    for candidate in candidates:
        if (candidate / "revision5-training-window-length-sensitivity.json").exists():
            return candidate
    raise FileNotFoundError("Revision 5 window-sensitivity aggregate JSON is missing.")


RESULTS = result_directory(ROOT)
FIGURES = ROOT / "research_artifacts" / "figures" if (ROOT / "research_artifacts").exists() else ROOT / "figures" / "revision3"


def main() -> None:
    payload = json.loads((RESULTS / "revision5-training-window-length-sensitivity.json").read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    labels = ["200", "500", "full\n(1,028)"]
    keys = ["window_200", "window_500", "full_available_history"]
    means = np.asarray([aggregate[key]["mean_roc_auc"] for key in keys])
    standard_deviations = np.asarray([aggregate[key]["standard_deviation_roc_auc"] for key in keys])
    differences = payload["paired_roc_auc_difference_vs_window_200"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    figure, axis = plt.subplots(figsize=(6.3, 3.0))
    positions = np.arange(len(keys))
    colors = ["#3B7897", "#76985A", "#C58A47"]
    bars = axis.bar(positions, means, yerr=standard_deviations, capsize=3.5, color=colors, edgecolor="#1A2730", linewidth=0.6, zorder=2)
    for bar, value, sd in zip(bars, means, standard_deviations):
        axis.text(bar.get_x() + bar.get_width() / 2, value + sd + 0.00018, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    axis.axhline(means[0], color="#3B7897", linestyle="--", linewidth=0.8, alpha=0.75, zorder=1)
    axis.text(1.96, means[0] + 0.00006, "200-window mean", ha="right", va="bottom", fontsize=6.8, color="#2D6179")
    axis.set_ylim(0.75, 0.78)
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Maximum training-history window (transitions)")
    axis.set_ylabel("Mean test ROC-AUC across 3 seeds")
    axis.set_title("Training-window sensitivity with full-history evaluation")
    axis.grid(axis="y", color="#D6DEE2", linewidth=0.65, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.text(0.01, -0.075, "Fixed student-disjoint split, DKT-64, AdamW (weight decay 0), batch size 64, eight-epoch cap and full-history test evaluation. The shared 0.75–0.78 scale avoids visually magnifying the small mean differences. Error bars are across-seed SDs; this is a descriptive sensitivity analysis, not a confidence interval or equivalence test.", fontsize=5.8)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"fig5_training_window_sensitivity.{suffix}", dpi=220, bbox_inches="tight")
    summary = {
        "window_500_minus_window_200_mean_auc": differences["window_500"]["mean_difference"],
        "full_available_history_minus_window_200_mean_auc": differences["full_available_history"]["mean_difference"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
