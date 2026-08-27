"""Reconstruct Fig. 1–Fig. 6 in a restrained Nature-style visual system.

Inputs are aggregate, de-identified JSON only. The script never reads raw learner
records, student identifiers, hidden states, checkpoints, or individual predictions.
Outputs are 600-dpi PNG plus SVG and PDF source-compatible exports.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure-data-nature-v31.json"
OUT = ROOT / "figures" / "nature-v31"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1B4F72"
TEAL = "#2A7F7F"
ORANGE = "#C65D3B"
GREY = "#6E7781"
LIGHT = "#E9EEF2"
DARK = "#20252B"
PALETTE = [NAVY, ORANGE, TEAL, GREY, "#8A6F9E", "#B07A50"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def finish(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    if grid:
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)


def fig1():
    fig, ax = plt.subplots(figsize=(7.1, 2.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    labels = ["Corrected\nASSISTments\nrecords", "Textualized\nprompt with\nexplicit fields", "Untouched base\nLM hidden states\nand output logits", "Representation\nreadability\n(silhouette + probe)", "Lexical response\nlabel audit\n(logit-score AUC)"]
    fills = ["#E8F0F4", "#F5E9DD", "#E8F0F4", "#F5E9DD", "#E8F0F4"]
    xs = np.linspace(0.9, 9.1, 5)
    for i, (x, label, fill) in enumerate(zip(xs, labels, fills)):
        box = FancyBboxPatch((x-0.85, 1.15), 1.7, 0.95, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=fill, edgecolor="#416A7A", linewidth=0.8)
        ax.add_patch(box); ax.text(x, 1.63, label, ha="center", va="center", linespacing=1.05, fontsize=6.2 if i == 3 else 7.2)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x+0.9,1.63),(xs[i+1]-0.9,1.63), arrowstyle="-|>", mutation_scale=9, linewidth=0.7, color="#416A7A"))
    ax.text(5, 0.48, "The two diagnostic branches are reported separately; neither is a future-response KT benchmark.", ha="center", va="center", fontsize=7)
    ax.text(5, 2.65, "Static prompt-interface audit workflow", ha="center", va="center", fontsize=9, fontweight="bold")
    finish(fig, "figure-01-static-prompt-interface-audit-workflow")


def fig2(d):
    x = np.array([d["permutation_accuracy_min"], d["permutation_accuracy_mean"], d["permutation_accuracy_max"]])
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.axhspan(x[0], x[2], color=LIGHT, zorder=0, label="Permutation range")
    ax.errorbar([0], [x[1]], yerr=[[x[1]-x[0]], [x[2]-x[1]]], fmt="o", color=GREY, capsize=3, markersize=4, linewidth=1.0, label="Permutation mean ± range")
    ax.scatter([1], [d["true_label_accuracy"]], s=28, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3, label="True-label accuracy")
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(0, 1.05)
    ax.set_xticks([0,1], ["Permuted labels", "True labels"])
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Random-label probe control")
    ax.text(1, d["true_label_accuracy"]+0.055, f'{d["true_label_accuracy"]:.3f}', ha="center", fontsize=7)
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)
    finish(fig, "figure-02-qwen-probe-permutation-control")


def fig3(d):
    matrix = np.asarray(d["matrix"])
    fig, ax = plt.subplots(figsize=(3.55, 3.25))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()), interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i,j]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6.2, color="white" if val > matrix.max()*0.55 else DARK)
    ax.set_xticks(range(len(d["labels"])), d["labels"], rotation=45, ha="right")
    ax.set_yticks(range(len(d["labels"])), d["labels"])
    ax.set_xlabel("Predicted skill ID"); ax.set_ylabel("True skill ID")
    ax.set_title("Qwen probe confusion matrix", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cbar.set_label("Records", rotation=90)
    cbar.outline.set_linewidth(0.4)
    clean_axes(ax, grid=False)
    finish(fig, "figure-03-qwen-probe-confusion-matrix")


def fig4(d):
    labels = ["Explicit", "Neutral", "Unknown", "Apple", "Table", "Removed"]
    keys = ["explicit", "neutral", "unknown", "apple", "table", "removed"]
    vals = [d[k]["auc"] for k in keys]; ps = [d[k]["p"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.1, 2.75))
    x = np.arange(len(labels)); bars = ax.bar(x, vals, color=PALETTE, width=0.68, edgecolor="none")
    ax.axhline(0.5, color=DARK, linewidth=0.65, linestyle=(0,(3,2)), label="AUC = 0.5")
    ax.set_ylim(0.25, 1.08); ax.set_ylabel("ROC AUC"); ax.set_xticks(x, labels)
    ax.set_title("Output sensitivity under the revised non-temporal prompt")
    for bar, v, p in zip(bars, vals, ps):
        label = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(bar.get_x()+bar.get_width()/2, max(v+0.035, 0.535), label, ha="center", va="bottom", fontsize=6.2)
    clean_axes(ax); ax.legend(frameon=False, loc="upper right")
    finish(fig, "figure-04-qwen-output-sensitivity")


def fig5(d):
    labels = ["Mean", "Max", "Last token"]
    keys = ["mean", "max", "last_token"]
    fig, ax = plt.subplots(figsize=(4.6, 2.55))
    rng = np.random.default_rng(42)
    for i,k in enumerate(keys):
        vals = np.asarray(d[k]); mean=vals.mean(); sd=vals.std(ddof=1)
        jitter = rng.uniform(-0.12,0.12,len(vals))
        ax.scatter(np.full(len(vals),i)+jitter, vals, s=16, color=NAVY if i<2 else ORANGE, edgecolor="white", linewidth=0.35, zorder=3)
        ax.errorbar(i, mean, yerr=sd, fmt="_", color=DARK, markersize=12, capsize=3, linewidth=1.0, zorder=4)
    ax.set_xticks(range(3), labels); ax.set_ylim(0.94,1.012); ax.set_ylabel("Student-level fold accuracy")
    ax.set_title("Student-level five-fold pooling ablation")
    clean_axes(ax)
    finish(fig, "figure-05-student-level-pooling-ablation")


def fig6(d):
    fig, axes = plt.subplots(1,3,figsize=(7.0,2.25), sharex=True)
    x=np.arange(3); labels=d["labels"]; width=0.32
    panels=[("Correctness mean",d["correctness"],"Correctness"), ("Unique students",d["students"],"Students"), ("Median order identifier",d["order_median"],"Median order")]
    for ax,(title,vals,ylabel) in zip(axes,panels):
        ax.bar(x-width/2,vals["deterministic"],width,color=GREY,label="Deterministic first-N")
        ax.bar(x+width/2,vals["random"],width,color=NAVY,label="Seed-42 random")
        ax.set_title(title); ax.set_xticks(x,labels); ax.set_ylabel(ylabel); clean_axes(ax)
        if title == "Median order identifier": ax.ticklabel_format(axis="y",style="sci",scilimits=(6,6),useMathText=True)
    axes[0].legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.02),ncol=2)
    fig.suptitle("Deterministic first-N versus stratified-random sample composition", y=1.04, fontsize=9, fontweight="bold")
    fig.tight_layout()
    finish(fig, "figure-06-sampling-distribution-comparison")


def main():
    d=json.loads(DATA.read_text(encoding="utf-8"))
    fig1(); fig2(d["fig2_probe"]); fig3(d["fig3_confusion"]); fig4(d["fig4_auc"]); fig5(d["fig5_pooling"]); fig6(d["fig6_sampling"])
    print(f"wrote six Nature-style figure triplets to {OUT}")

if __name__ == "__main__": main()
