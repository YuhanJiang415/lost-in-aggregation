"""Nature-style Figure for Experiment 1: format ranking robustness.

Claim: Structured textual inputs (Words/Coordinate) outperform visual/grid
inputs (Map/Picture) across all maze scales and tested LLMs.

Panels (183 mm x 75 mm, three columns):
  (a) SR by model x format, grouped bars + 1 SEM (n=150 per bar).
  (b) SR vs maze size, line plot per format, faceted by model + 1 SEM.
  (c) VMR by model x format, grouped bars + 1 SEM.

Usage:
    python -m analysis.exp1_format_ranking_fig
    python -m analysis.exp1_format_ranking_fig --run-dir results/exp1/20260410_170537
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct execution (`python analysis/exp1_format_ranking_fig.py`) as well
# as module execution (`python -m analysis.exp1_format_ranking_fig`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.load_results import (
    PROJECT_ROOT,
    find_latest_run,
    load_experiment_dir,
)

FORMATS = ["words", "coordinate", "map", "picture"]
FORMAT_LABELS = {
    "words": "Words",
    "coordinate": "Coordinate",
    "map": "Map",
    "picture": "Picture",
}
# Cool family (deep -> light): structured text.
# Warm family (light -> deep): visual.
FORMAT_COLORS = {
    "words":      "#2F4F7A",
    "coordinate": "#6F9CC2",
    "map":        "#E3B884",
    "picture":    "#C2725A",
}
SIZES = [3, 5, 7]
MODELS_ORDER = ["gpt-4o", "deepseek-v3", "llama-3.3-70b"]
MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "deepseek-v3": "DeepSeek-V3",
    "llama-3.3-70b": "Llama-3.3-70B",
}

MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    })


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bernoulli_sem(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Mean and Wilson 95% confidence interval for a Bernoulli proportion.

    Returns (p_hat, lo, hi). Wilson (not +/- 1 SEM) is used because the SEM
    degenerates to zero width at p=0 and p=1 -- exactly the saturated cells
    that dominate this benchmark -- whereas Wilson stays non-degenerate and
    gives proper 95% coverage. Name kept for call-site compatibility.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    # Symmetric Wilson half-width around p (matches fig2/fig3 bar style);
    # stays non-degenerate at p=0/1 unlike the +/- 1 SEM it replaces.
    return p, max(0.0, p - half), min(1.0, p + half)


def mean_sem(x: np.ndarray) -> Tuple[float, float, float]:
    """Mean and +/- 1 SEM band for a continuous sample."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0
    return mean, mean - sem, mean + sem


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> dict:
    """Compute SR/VMR aggregates with CIs."""
    sr_col = "metrics.overall.SR"
    vmr_col = "metrics.overall.VMR"

    # Pooled across size: (model, format) -> stats
    sr_pooled = {}
    vmr_pooled = {}
    for (model, fmt), grp in df.groupby(["model", "format"]):
        sr_vals = grp[sr_col].astype(bool).to_numpy()
        n = sr_vals.size
        k = int(sr_vals.sum())
        sr_pooled[(model, fmt)] = bernoulli_sem(k, n) + (n,)

        vmr_vals = grp[vmr_col].dropna().astype(float).to_numpy()
        vmr_pooled[(model, fmt)] = mean_sem(vmr_vals)

    # Per-size: (model, format, size) -> SR with SEM band
    sr_by_size = {}
    for (model, fmt, size), grp in df.groupby(["model", "format", "effective_size"]):
        sr_vals = grp[sr_col].astype(bool).to_numpy()
        n = sr_vals.size
        k = int(sr_vals.sum())
        sr_by_size[(model, fmt, int(size))] = bernoulli_sem(k, n) + (n,)

    return {"sr_pooled": sr_pooled, "vmr_pooled": vmr_pooled, "sr_by_size": sr_by_size}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _grouped_bars(
    ax: plt.Axes,
    stats: dict,
    metric_label: str,
    panel_letter: str,
    show_y_label: bool = True,
) -> None:
    """Grouped bars: x = model, hue = format, y = metric mean with CI."""
    n_models = len(MODELS_ORDER)
    n_fmt = len(FORMATS)
    bar_w = 0.78 / n_fmt
    x = np.arange(n_models)

    for j, fmt in enumerate(FORMATS):
        means, lo_err, hi_err = [], [], []
        for model in MODELS_ORDER:
            m, lo, hi, *_ = stats[(model, fmt)]
            means.append(m)
            lo_err.append(m - lo)
            hi_err.append(hi - m)
        offsets = x - 0.39 + (j + 0.5) * bar_w
        ax.bar(
            offsets, means, bar_w * 0.92,
            color=FORMAT_COLORS[fmt],
            edgecolor="white", linewidth=0.4,
            label=FORMAT_LABELS[fmt],
        )
        ax.errorbar(
            offsets, means,
            yerr=[lo_err, hi_err],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.6, capsize=1.4, capthick=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in MODELS_ORDER],
        rotation=18, ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    if show_y_label:
        ax.set_ylabel(metric_label)
    ax.tick_params(axis="x", pad=1)
    ax.text(-0.20, 1.06, panel_letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_a_sr(ax: plt.Axes, agg: dict) -> None:
    _grouped_bars(ax, agg["sr_pooled"], "Success rate", "a")
    ax.set_title("SR pooled over scales", pad=2)


def panel_c_vmr(ax: plt.Axes, agg: dict) -> None:
    _grouped_bars(ax, agg["vmr_pooled"], "Valid-move ratio", "c")
    ax.set_title("VMR pooled over scales", pad=2)


def panel_b_scale(fig, gs_cell, agg: dict) -> None:
    """3 small subplots inside the middle column, one per model.

    Lines: format. X: maze effective size. Y: SR with +/- 1 SEM band
    (consistent with fig2_exp2_main).
    """
    inner = gs_cell.subgridspec(1, len(MODELS_ORDER), wspace=0.18)
    axes = []
    for i, model in enumerate(MODELS_ORDER):
        ax = fig.add_subplot(inner[0, i])
        axes.append(ax)
        for fmt in FORMATS:
            means, los, his = [], [], []
            for s in SIZES:
                m, lo, hi, *_ = agg["sr_by_size"][(model, fmt, s)]
                means.append(m); los.append(lo); his.append(hi)
            color = FORMAT_COLORS[fmt]
            ax.plot(
                SIZES, means,
                marker="o", linewidth=1.0, markersize=2.8,
                color=color, label=FORMAT_LABELS[fmt], zorder=3,
            )
            ax.fill_between(
                SIZES, los, his,
                color=color, alpha=0.18, linewidth=0, zorder=2,
            )
        ax.set_xticks(SIZES)
        ax.set_xticklabels([f"{s}x{s}" for s in SIZES])
        ax.set_ylim(-0.02, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(MODEL_LABELS[model], fontsize=7.5, fontweight="bold", pad=3)
        if i == 0:
            ax.set_ylabel("Success rate")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Maze size", labelpad=1)

    axes[0].text(-0.50, 1.06, "b", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Compose figure
# ---------------------------------------------------------------------------

def build_figure(agg: dict) -> plt.Figure:
    fig_w = 183 / MM_PER_INCH
    fig_h = 75 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 1.45, 1.0],
        wspace=0.34,
        left=0.055, right=0.985,
        top=0.80, bottom=0.21,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a_sr(ax_a, agg)

    panel_b_scale(fig, gs[0, 1], agg)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_c_vmr(ax_c, agg)

    # Single legend at top
    handles = [
        mpl.patches.Patch(
            facecolor=FORMAT_COLORS[f], edgecolor="white", label=FORMAT_LABELS[f]
        )
        for f in FORMATS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4, frameon=False,
        handlelength=1.1, handleheight=0.9,
        columnspacing=1.6, fontsize=7,
    )
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_pub(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--out", type=str,
        default="paper/figures/rq1_format_robust",
        help="Output stem (relative to project root)",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    else:
        run_dir = find_latest_run("exp1")
    print(f"Using run: {run_dir}")

    df = load_experiment_dir(run_dir)
    # Filter to the canonical conditions only
    df = df[df["format"].isin(FORMATS) & df["model"].isin(MODELS_ORDER)]
    df = df[df["effective_size"].isin(SIZES)]
    print(f"Loaded {len(df)} trials "
          f"({df.groupby(['model','format','effective_size']).size().min()}-"
          f"{df.groupby(['model','format','effective_size']).size().max()} per cell)")

    agg = aggregate(df)
    setup_style()
    fig = build_figure(agg)

    out = PROJECT_ROOT / args.out
    save_pub(fig, out)
    print(f"Saved {out}.{{svg,pdf,png,tiff}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
