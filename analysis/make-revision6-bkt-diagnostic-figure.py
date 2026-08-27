"""Render an aggregate-safe supplementary figure for Revision 6 BKT diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if HERE.parents[1].name == "code" else HERE.parents[1]
RESULTS = ROOT / "research_code" / "results" / "revision3" if (ROOT / "research_code").exists() else ROOT / "results" / "revision3"
FIGURES = ROOT / "research_artifacts" / "figures" if (ROOT / "research_artifacts").exists() else ROOT / "figures" / "revision3"


def main() -> None:
    audit = json.loads((RESULTS / "revision6-statistical-and-bkt-audits.json").read_text(encoding="utf-8"))
    diagnostics = audit["bkt_diagnostics"]
    sensitivity = audit["bkt_iteration_sensitivity"]
    quantiles = diagnostics["parameter_quantiles_across_skills"]
    names = ["P(L0)", "P(T)", "P(G)", "P(S)"]
    medians = np.asarray([quantiles[name]["p50"] for name in names])
    low = np.asarray([quantiles[name]["p05"] for name in names])
    high = np.asarray([quantiles[name]["p95"] for name in names])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 0.8]})

    axis = axes[0]
    positions = np.arange(len(names))
    axis.errorbar(positions, medians, yerr=np.vstack([medians - low, high - medians]), fmt="o", color="#3B7897", ecolor="#3B7897", capsize=4, linewidth=1.1, markersize=5, zorder=3)
    axis.scatter(positions, low, marker="_", color="#3B7897", s=70, zorder=3)
    axis.scatter(positions, high, marker="_", color="#3B7897", s=70, zorder=3)
    axis.set_xticks(positions, names)
    axis.set_ylim(0.0, 0.55)
    axis.set_ylabel("Parameter estimate")
    axis.set_title("Across-skill BKT parameter summaries")
    axis.grid(axis="y", color="#D6DEE2", linewidth=0.65, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(0.02, 0.97, "points: median; bars: P05–P95\n149 atomic skills; 20 fixed EM iterations", transform=axis.transAxes, va="top", fontsize=6.7)

    axis = axes[1]
    proportions = np.asarray([
        diagnostics["final_iteration_max_absolute_parameter_change"]["proportion_at_or_below_threshold"],
        sensitivity["fixed_100_final_iteration_max_absolute_parameter_change"]["proportion_at_or_below_threshold"],
    ])
    labels = ["20 iterations\n(19→20)", "100 iterations\n(99→100)"]
    bars = axis.bar(np.arange(2), proportions, color=["#C58A47", "#76985A"], edgecolor="#1A2730", linewidth=0.6)
    for bar, value in zip(bars, proportions):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value * 100:.1f}%", ha="center", va="bottom")
    axis.set_ylim(0.0, 0.90)
    axis.set_xticks(np.arange(2), labels)
    axis.set_ylabel("Skills with final max |Δparameter| ≤ 1e−5")
    axis.set_title("Fixed-iteration stability proxy")
    axis.grid(axis="y", color="#D6DEE2", linewidth=0.65, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.text(0.01, -0.09, "All values are aggregate across skills. Final-update change is a fixed-iteration stability proxy, not a likelihood convergence certificate, global optimum proof or multi-start diagnostic. Per-skill parameters are not released.", fontsize=6.0)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"figA2-bkt-fit-diagnostics.{suffix}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
