"""Supplementary Figure 1: by-question-subtype accuracy curves.

Claim: Across the 8 question subtypes, individual probes follow distinct
scaling trajectories. Pure single-dimension probes (Fine, Meso, Macro)
remain above chance at all scales; the Meso × Macro coupled probe
(junction_branch) collapses fastest of all; the NonSpatial control band
reveals that row_listing is trivially solved at all scales while
cell_count collapses from near-ceiling at 3x3 to near-chance by 10x10.

Panels (183 mm x 75 mm, five facets in a row):
  (a) Fine         -- passable_directions
  (b) Meso         -- dead_end_recognition (yes/no)
  (c) Macro        -- initial_direction, midcourse_direction (both 4-choice)
  (d) Meso × Macro -- junction_branch (4-choice; drawn in the coupling
                      colour #9C3E3E, matching Fig 2 panel b/c and Supp Fig 2)
  (e) NonSpatial   -- cell_count, wall_stretch, row_listing

Source: results/exp2_independent/20260516_medium/summary.json (by_question_type)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MM_PER_INCH = 25.4

INDEP_SUMMARY = (PROJECT_ROOT / "exp_output" / "exp2_independent"
                 / "medium" / "summary.json")

SIZES = [3, 5, 7, 10, 15, 20, 30]

# Use the SAME dimension colors as Fig 1/2/4 for cross-figure consistency.
# Coupling colour for the Meso × Macro dedicated panel (junction_branch).
# Kept in sync with Fig 2 panel b/c and Supp Fig 2 — see captions.md
# "Cross-figure conventions: Coupling colour".
COUPLING_COLOR = "#9C3E3E"

DIM_COLORS = {
    "fine":       "#4F8FCB",
    "meso":       "#E2A24A",
    "macro":      "#7A5BA0",
    "meso_macro": COUPLING_COLOR,
    "nonspatial": "#6E6E6E",
}
DIM_LABELS = {
    "fine":       "Fine",
    "meso":       "Meso",
    "macro":      "Macro",
    "meso_macro": "Meso × Macro",
    "nonspatial": "NonSpatial",
}

# Each subtype: (parent dim, short label for legend, line style, chance level).
SUBTYPES: Dict[str, Tuple[str, str, str, float]] = {
    "passable_directions":   ("fine",       "Passable directions",   "-",     0.25),
    "junction_branch":       ("meso",       "Junction branch (4-c.)", "-",    0.25),
    "dead_end_recognition":  ("meso",       "Dead-end (yes/no)",     "-",     0.50),
    "initial_direction":     ("macro",      "Initial direction",     "-",     0.25),
    "midcourse_direction":   ("macro",      "Midcourse direction",   "--",    0.25),
    "cell_count":            ("nonspatial", "Cell count",            "-",     0.25),
    "wall_stretch":          ("nonspatial", "Wall stretch",          "--",    0.25),
    "row_listing":           ("nonspatial", "Row listing (free)",    ":",     0.0),
}
SUBTYPES_BY_DIM: Dict[str, List[str]] = {
    "fine":       ["passable_directions"],
    "meso":       ["dead_end_recognition"],
    "macro":      ["midcourse_direction",  "initial_direction"],
    # Dedicated panel for the Meso × Macro coupled probe; drawn in
    # COUPLING_COLOR so it visually matches Fig 2 panel b/c and Supp Fig 2.
    "meso_macro": ["junction_branch"],
    "nonspatial": ["cell_count", "wall_stretch", "row_listing"],
}
N_FULL = 50   # threshold under which a per-cell point is "exploratory"


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
        "lines.markersize": 3.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    })


def bernoulli_sem(p: float, n: int, z: float = 1.96) -> float:
    """Wilson 95% half-width for a Bernoulli proportion (symmetric approx).

    Replaces the +/- 1 SEM half-width for consistency with the main figures.
    Name kept for call-site compatibility.
    """
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    return (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_subtype_by_size() -> Dict[Tuple[str, int], Tuple[float, float, int]]:
    """{(subtype, size): (acc, SEM, n_total)} pooled across all 3 models."""
    summary = json.load(open(INDEP_SUMMARY))
    agg: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
    for v in summary.values():
        sz = int(v["effective_size"].lstrip("s"))
        if sz not in SIZES:
            continue
        for qt, qstats in v.get("by_question_type", {}).items():
            agg[(qt, sz)].append((qstats["n_correct"], qstats["n"]))
    out: Dict[Tuple[str, int], Tuple[float, float, int]] = {}
    for (qt, sz), items in agg.items():
        n = sum(t for _, t in items)
        k = sum(c for c, _ in items)
        if n == 0:
            continue
        p = k / n
        out[(qt, sz)] = (p, bernoulli_sem(p, n), n)
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_subtype_line(ax, subtype: str, data, color: str):
    _dim, label, style, _chance = SUBTYPES[subtype]
    full_xs, full_ys, full_se = [], [], []
    expl_xs, expl_ys = [], []
    for sz in SIZES:
        row = data.get((subtype, sz))
        if row is None:
            continue
        acc, sem, n = row
        if n >= N_FULL:
            full_xs.append(sz); full_ys.append(acc); full_se.append(sem)
        else:
            expl_xs.append(sz); expl_ys.append(acc)
    # Combined line uses ALL points (full + exploratory) so it stays connected.
    all_pts = sorted(
        [(s, data[(subtype, s)][0]) for s in SIZES if (subtype, s) in data],
        key=lambda t: t[0],
    )
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    ax.plot(xs, ys, linestyle=style, color=color, linewidth=1.0,
            label=label, zorder=3)
    # Filled markers + SEM band on the full-n points only
    if full_xs:
        ax.plot(full_xs, full_ys, "o", color=color, markersize=3.0,
                markerfacecolor=color, markeredgecolor=color, zorder=4)
        ax.fill_between(full_xs,
                        np.array(full_ys) - np.array(full_se),
                        np.array(full_ys) + np.array(full_se),
                        color=color, alpha=0.18, linewidth=0, zorder=2)
    # Hollow markers on exploratory (small-n) points
    if expl_xs:
        ax.scatter(expl_xs, expl_ys, marker="o", s=18,
                   facecolors="white", edgecolors=color, linewidths=0.8,
                   zorder=4)


def _draw_chance_lines(ax, levels: List[float]):
    """Draw faint dashed reference lines at given chance levels (0..1)."""
    seen = set()
    for lv in levels:
        key = round(lv, 3)
        if key in seen or key <= 0:
            continue
        seen.add(key)
        ax.axhline(lv, color="#999", linewidth=0.5, linestyle=":", zorder=1)
        # tiny on-line label far right
        ax.text(SIZES[-1] - 0.4, lv + 0.012, f"chance {int(lv*100)}%",
                fontsize=5.0, color="#777", ha="right", va="bottom")


def panel_dim(ax, dim: str, data, panel_letter: str,
              show_y_label: bool):
    color = DIM_COLORS[dim]
    subtypes = SUBTYPES_BY_DIM[dim]
    chance_levels = sorted({SUBTYPES[s][3] for s in subtypes if SUBTYPES[s][3] > 0})
    _draw_chance_lines(ax, chance_levels)
    for subtype in subtypes:
        _plot_subtype_line(ax, subtype, data, color)

    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] - 1, SIZES[-1] + 1)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Maze effective size")
    if show_y_label:
        ax.set_ylabel("MCQ accuracy")
    ax.set_title(DIM_LABELS[dim], pad=2)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.2,
              fontsize=5.4, handleheight=0.8)
    ax.text(-0.18, 1.06, panel_letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_figure(data):
    fig_w = 183 / MM_PER_INCH
    fig_h = 75 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=5,
        wspace=0.32,
        left=0.05, right=0.99,
        top=0.88, bottom=0.20,
    )
    panels = [
        ("fine",       "a"),
        ("meso",       "b"),
        ("macro",      "c"),
        ("meso_macro", "d"),
        ("nonspatial", "e"),
    ]
    for i, (dim, letter) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        panel_dim(ax, dim, data, letter, show_y_label=(i == 0))
    return fig


def save_pub(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str,
                        default="paper/figures/figs1_question_subtypes")
    args = parser.parse_args()

    setup_style()
    data = load_subtype_by_size()
    print("Loaded subtypes:")
    for qt in SUBTYPES:
        for sz in SIZES:
            row = data.get((qt, sz))
            if row:
                print(f"  {qt:<22} s{sz:<3} acc={row[0]:.3f}  SEM={row[1]:.3f}  n={row[2]}")

    fig = build_figure(data)
    out = PROJECT_ROOT / args.out
    save_pub(fig, out)
    print(f"Saved {out}.{{svg,pdf,png,tiff}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
