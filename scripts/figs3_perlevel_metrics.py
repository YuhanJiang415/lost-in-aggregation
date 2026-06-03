"""Supplementary figure: per-level diagnostic metrics inside one-shot navigation.

Claim: Inside the navigation loop the three cognitive levels degrade
differently with maze size.
  (a) Fine: erosion is driven by teleports (TR climbs), not wall hits (WCR
      stays low), so positioning accuracy (PA) falls.
  (b) Meso: junction accuracy (JA) drops steeply; the model enters dead ends
      (DER) but its backtrack-success rate (BSR) stays near zero, i.e. it
      almost never recovers from a dead end on its own.
  (c) Macro: the most stable level: the windowed progress rate (MPR) declines
      only modestly and the direction-drift rate (DDR) stays low.

Data: exp_output/exp2_onehot/results.csv, medium difficulty, pooled over the
three models (GPT-4o, DeepSeek-V3, Llama-3.3-70B), Coordinate input.
Bands: 95% CI (+/- 1.96 SEM across trials).

Mirrors Table tab:rq2-perlevel in the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MM_PER_INCH = 25.4

RESULTS_CSV = PROJECT_ROOT / "exp_output" / "exp2_onehot" / "results.csv"
DEFAULT_OUT = (PROJECT_ROOT / "69fca4a0e192a8e6364e86c6" / "figures"
               / "appendix_perlevel_metrics")

SIZES = [3, 5, 7, 10, 15, 20, 30]

# (level title, level colour, [(metric, label, linestyle, marker), ...])
# Colours match the Fine/Meso/Macro identity used across the paper's figures.
PANELS: List[Tuple[str, str, List[Tuple[str, str, str, str]]]] = [
    ("Fine", "#4F8FCB", [
        ("fine.PA",  "PA (positioning acc.)", "-",  "o"),
        ("fine.TR",  "TR (teleport)",         "--", "s"),
        ("fine.WCR", "WCR (wall collision)",  ":",  "^"),
    ]),
    ("Meso", "#E2A24A", [
        ("meso.JA",  "JA (junction acc.)",    "-",  "o"),
        ("meso.DER", "DER (dead-end entry)",  "--", "s"),
        ("meso.BSR", "BSR (backtrack succ.)", ":",  "^"),
    ]),
    ("Macro", "#7A5BA0", [
        ("macro.MPR", "MPR (progress rate)",  "-",  "o"),
        ("macro.DDR", "DDR (direction drift)", "--", "s"),
    ]),
]

PANEL_LETTERS = ["a", "b", "c"]


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
        "legend.fontsize": 5.4,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    })


def shade(hex_color: str, factor: float) -> str:
    """Lighten (factor>0) a hex colour toward white by the given fraction."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def metric_by_size(df: pd.DataFrame, col: str):
    """Mean and 95% CI half-width (%) of a metric by size, pooled over models."""
    means, cis = [], []
    for sz in SIZES:
        s = df.loc[df["effective_size"] == sz, col].dropna()
        n = len(s)
        m = s.mean() * 100 if n else np.nan
        ci = (1.96 * s.std(ddof=1) / np.sqrt(n) * 100) if n > 1 else 0.0
        means.append(m)
        cis.append(ci)
    return np.array(means), np.array(cis)


def panel(ax, df, title, color, metrics, letter):
    # Distinct, readable shades within the level's colour family.
    shades = [0.0, 0.34, 0.6][: len(metrics)]
    for (col, label, style, marker), sh in zip(metrics, shades):
        c = shade(color, sh)
        m, ci = metric_by_size(df, col)
        ax.plot(SIZES, m, linestyle=style, marker=marker, color=c,
                linewidth=1.2, markersize=3.0, label=label, zorder=4)
        ax.fill_between(SIZES, m - ci, m + ci, color=c, alpha=0.15,
                        linewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.minorticks_off()
    ax.set_xlim(SIZES[0] - 0.3, SIZES[-1] + 4)
    ax.set_ylim(-3, 103)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Maze effective size")
    if letter == "a":
        ax.set_ylabel("Rate / accuracy (%)")
    ax.set_title(title, pad=3, color=color)
    ax.legend(loc="upper right", handlelength=1.8, borderaxespad=0.2,
              labelspacing=0.3, handletextpad=0.5)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def build_figure(df):
    fig_w = 183 / MM_PER_INCH
    fig_h = 62 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.22,
                          left=0.07, right=0.99, top=0.90, bottom=0.18)
    for i, (title, color, metrics) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        panel(ax, df, title, color, metrics, PANEL_LETTERS[i])
    return fig


def save_pub(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    setup_style()
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["difficulty"] == "medium"].copy()
    print(f"medium trials: {len(df)} | models: {df['model'].nunique()} "
          f"| formats: {sorted(df['format'].unique())}")
    print("Per-metric means by size (%):")
    for _title, _c, metrics in PANELS:
        for col, label, *_ in metrics:
            m, _ci = metric_by_size(df, col)
            print(f"  {col:10s} " + " ".join(f"{v:4.0f}" for v in m))

    fig = build_figure(df)
    save_pub(fig, Path(args.out))
    print(f"Saved {args.out}.{{png,svg,pdf}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
