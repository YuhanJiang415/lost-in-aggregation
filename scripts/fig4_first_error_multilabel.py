"""Figure 4 (multi-label version): first-error composition under the 8-label
schema where one FSE may fire several labels simultaneously.

Claim: First-step errors are dominated by Meso topological choices (especially
"start_junction_wrong") and Fine perception failures (teleport, wall, plus
position hallucination in large mazes), while pure global-direction failure
is rare. 18% of FSEs co-fire across Fine and Meso, showing that local-execution
and topological-choice failures often surface together.

Panels (150 mm x 170 mm):
  (a) Ternary scatter: each (size / model / difficulty) condition projected
      into Fine-Meso-Macro barycentric coords using per-step-normalised
      averaging (each FSE distributes mass 1 across the dimensions it
      activates, then average over FSEs in the group).
  (b) Multi-label bar chart: marginal share of each of the 8 labels over
      the FSE pool. Bars are coloured by dimension and shaded within each
      dimension to disambiguate sub-labels. Note: shares do NOT sum to 1
      (multi-hot labelling).

Source: results/exp2_oneshot/20260421_232155/first_error_multilabel/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.patches import Rectangle

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MM_PER_INCH = 25.4

FEM_DIR = (PROJECT_ROOT / "results" / "exp2_oneshot" / "20260421_232155"
           / "first_error_multilabel")

DIMS = ["fine", "meso", "macro"]
DIM_LABELS = {"fine": "Fine", "meso": "Meso", "macro": "Macro"}
DIM_COLORS = {
    "fine":  "#4F8FCB",
    "meso":  "#E2A24A",
    "macro": "#7A5BA0",
}

# Within-dimension shading. Order matches the LABEL_ORDER below: lighter shade
# for less-frequent / specialised labels, darker shade for primary labels.
LABEL_COLORS = {
    # Fine — blues
    "fine.teleport":               "#2F6FA8",  # darkest fine
    "fine.wall":                   "#5F95C8",
    "fine.wrong_start_position":   "#7DACD4",
    "fine.revisit":                "#A5C5DF",
    "fine.empty_or_unparsed_path": "#C5DCEB",  # lightest fine
    # Meso — oranges
    "meso.start_junction_wrong":   "#C77F2A",  # darkest meso
    "meso.t_junction_wrong":       "#E2A24A",
    "meso.x_junction_wrong":       "#EBC07A",  # lighter meso
    # Macro — purple
    "macro.wrong_big_direction":   "#6B4E94",
}

LABEL_DIM = {
    "fine.teleport":               "fine",
    "fine.wall":                   "fine",
    "fine.revisit":                "fine",
    "fine.wrong_start_position":   "fine",
    "fine.empty_or_unparsed_path": "fine",
    "meso.t_junction_wrong":       "meso",
    "meso.x_junction_wrong":       "meso",
    "meso.start_junction_wrong":   "meso",
    "macro.wrong_big_direction":   "macro",
}

# Sequential colormap per dimension: light = small maze, dark = large maze.
DIM_CMAPS = {"fine": "Blues", "meso": "Oranges", "macro": "Purples"}


def _size_shade(dim: str, sz_idx: int, n_sizes: int) -> tuple:
    """Color in `dim`'s sequential cmap, light → dark across sizes.

    `sz_idx` is 0..n_sizes-1 (s3 → s30 in this study).
    """
    cmap = colormaps[DIM_CMAPS[dim]]
    # Sample at 0.25 (lightest) to 0.92 (darkest) so neither end is washed-out
    # nor pure-black; keeps the 7-step gradient legible at small sizes.
    t = 0.25 + 0.67 * (sz_idx / max(1, n_sizes - 1))
    return cmap(t)

# Bar display labels (short, human-readable)
LABEL_DISPLAY = {
    "fine.teleport":               "Teleport",
    "fine.wall":                   "Wall hit",
    "fine.revisit":                "Stay / back-step",
    "fine.wrong_start_position":   "Hallucinated start",
    "fine.empty_or_unparsed_path": "Empty / unparsed",
    "meso.t_junction_wrong":       "T-junction wrong",
    "meso.x_junction_wrong":       "X-junction wrong",
    "meso.start_junction_wrong":   "Start junction wrong",
    "macro.wrong_big_direction":   "Wrong direction",
}

# Display order in the bar panel: by descending overall share is computed at
# runtime, but we group within dimensions for visual coherence.
FINE_LABELS = [
    "fine.teleport", "fine.wall", "fine.wrong_start_position",
    "fine.revisit", "fine.empty_or_unparsed_path",
]
MESO_LABELS = [
    "meso.start_junction_wrong", "meso.t_junction_wrong",
    "meso.x_junction_wrong",
]
MACRO_LABELS = ["macro.wrong_big_direction"]
ALL_LABELS = FINE_LABELS + MESO_LABELS + MACRO_LABELS
DIM_PRESENT = {"fine": "fine_present",
               "meso": "meso_present",
               "macro": "macro_present"}

# The full triangle is the hero of panel (a). A small trapezoid inset placed in
# the upper interior shows the dense bottom strip (Macro ∈ [0, INSET_MACRO_MAX])
# with the data points spread out for legibility.
INSET_MACRO_MAX = 0.05
# Visible Meso range for the inset (Fine is its complement at Macro≈0).
# Data Meso lives in ≈ [0.47, 0.73]; this tighter window stops the inset from
# wasting space on Meso 0–40% and 80–100% where no data lives.
INSET_MESO_RANGE = (0.30, 0.85)
# Vertical height of the inset trapezoid in DATA coords. The full triangle uses
# sqrt(3)/2 ≈ 0.866; making this smaller squashes the Macro axis so the inset
# reads as a landscape parallelogram instead of a tall portrait one.
INSET_Y_HEIGHT = 0.42

# === Panel-a sub-column width ratios ===
# Panel a is split into three sub-columns: inset (left) | triangle (middle) |
# legend (right). Adjust the ratios to enlarge/shrink any of the three.
# All three share the same vertical span (panel-a row of the outer gridspec).
PANEL_A_WIDTH_RATIOS = (1.40, 2.60, 0.90)  # inset / triangle / legend

SIZE_ORDER = [3, 5, 7, 10, 15, 20, 30]
MODEL_ORDER = ["gpt-4o", "deepseek-v3", "llama-3.3-70b"]
MODEL_LABELS = {
    "gpt-4o":        "GPT-4o",
    "deepseek-v3":   "DeepSeek-V3",
    "llama-3.3-70b": "Llama-3.3-70B",
}
MODEL_COLORS = {
    "gpt-4o":        "#2F5F8B",
    "deepseek-v3":   "#C2725A",
    "llama-3.3-70b": "#6F9CC2",
}
DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_COLORS = {
    "easy":   "#3F8E3A",
    "medium": "#D6A33A",
    "hard":   "#B83A3A",
}


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


# ---------------------------------------------------------------------------
# Ternary geometry (Fine bot-L, Meso bot-R, Macro top)
# ---------------------------------------------------------------------------

def tern_to_xy(fine: float, meso: float, macro: float,
                macro_max: float = 1.0,
                y_height: float | None = None) -> tuple:
    """Ternary projection.

    With macro_max < 1 the y-axis is stretched 1/macro_max so a small Macro
    range fills the panel height (used by the inset). `y_height` sets the
    full vertical extent of the projection in data coords (default
    sqrt(3)/2 = the natural equilateral-triangle height). Pass a smaller
    value to render Macro as a squashed (landscape) axis.
    """
    x = meso + 0.5 * macro
    h_factor = y_height if y_height is not None else np.sqrt(3) / 2
    y = h_factor * (macro / macro_max)
    return x, y


def draw_ternary_axes(ax) -> None:
    h = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="#333",
            linewidth=0.8, zorder=2)

    for t in (0.25, 0.50, 0.75):
        p1 = tern_to_xy(t, 1 - t, 0)
        p2 = tern_to_xy(t, 0, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#dcdcdc", linewidth=0.4, zorder=0)
        p1 = tern_to_xy(1 - t, t, 0)
        p2 = tern_to_xy(0, t, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#dcdcdc", linewidth=0.4, zorder=0)
        p1 = tern_to_xy(1 - t, 0, t)
        p2 = tern_to_xy(0, 1 - t, t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#dcdcdc", linewidth=0.4, zorder=0)

    ax.text(-0.04, -0.04, "Fine\n100%", ha="right", va="top",
            color=DIM_COLORS["fine"], fontweight="bold", fontsize=7.2)
    ax.text(1.04, -0.04, "Meso\n100%", ha="left", va="top",
            color=DIM_COLORS["meso"], fontweight="bold", fontsize=7.2)
    ax.text(0.5, h + 0.04, "Macro 100%", ha="center", va="bottom",
            color=DIM_COLORS["macro"], fontweight="bold", fontsize=7.2)

    for t in (0.25, 0.5, 0.75):
        x, y = tern_to_xy(1 - t, t, 0)
        ax.text(x, y - 0.025, f"{int(t*100)}", ha="center", va="top",
                fontsize=5.4, color="#666")
        x, y = tern_to_xy(1 - t, 0, t)
        ax.text(x - 0.018, y + 0.005, f"{int((1-t)*100)}",
                ha="right", va="bottom", fontsize=5.4, color="#666", rotation=60)
        x, y = tern_to_xy(0, 1 - t, t)
        ax.text(x + 0.018, y + 0.005, f"{int(t*100)}",
                ha="left", va="bottom", fontsize=5.4, color="#666", rotation=-60)

    bx, by = tern_to_xy(0.5, 0.5, 0.0)
    ax.text(bx, by - 0.085, "Meso (%)", ha="center", va="top",
            fontsize=6.4, color="#333", fontweight="bold")
    edge_offset = 0.135
    half_sqrt3 = np.sqrt(3) / 2
    lx, ly = tern_to_xy(0.5, 0, 0.5)
    ax.text(lx - edge_offset * half_sqrt3, ly + edge_offset * 0.5,
            "Fine (%)", ha="center", va="center",
            rotation=60, fontsize=6.4, color="#333", fontweight="bold")
    rx, ry = tern_to_xy(0, 0.5, 0.5)
    ax.text(rx + edge_offset * half_sqrt3, ry + edge_offset * 0.5,
            "Macro (%)", ha="center", va="center",
            rotation=-60, fontsize=6.4, color="#333", fontweight="bold")

    cx, cy = tern_to_xy(1 / 3, 1 / 3, 1 / 3)
    ax.scatter([cx], [cy], marker="+", color="#888", s=22, linewidth=0.6,
               zorder=3)
    ax.text(cx, cy + 0.025, "no-preference\nbaseline",
            ha="center", va="bottom",
            fontsize=5.2, color="#888", style="italic", linespacing=1.0)

    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.16, h + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_inset_trapezoid(ax_inset, macro_max: float,
                          meso_range: tuple = (0.0, 1.0),
                          y_height: float | None = None) -> None:
    """Draw the zoomed-in trapezoid on `ax_inset`.

    - Macro is zoomed to [0, macro_max]   (y-axis stretch).
    - Meso (= x-axis at Macro=0) is windowed to `meso_range`. Fine is the
      complement so the bottom edge double-labels (Meso below, Fine above).
    - `y_height` is the vertical extent in data coords (default sqrt(3)/2 for
      the natural equilateral aspect; use a smaller value for a landscape
      inset that squashes the Macro axis).
    """
    h = y_height if y_height is not None else np.sqrt(3) / 2
    m_lo, m_hi = meso_range

    # Outline: bottom-left, bottom-right, top-right, top-left, close
    tl_x = m_lo + 0.5 * macro_max
    tr_x = m_hi + 0.5 * macro_max
    ax_inset.plot([m_lo, m_hi, tr_x, tl_x, m_lo],
                  [0, 0, h, h, 0],
                  color="#333", linewidth=0.7, zorder=2)

    # Pick Meso grid lines that fall inside the visible window.
    candidate_ticks = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    meso_ticks = [t for t in candidate_ticks if m_lo + 0.02 < t < m_hi - 0.02]
    # Constant-Meso grid lines (slanted)
    for m in meso_ticks:
        ax_inset.plot([m, m + 0.5 * macro_max], [0, h],
                      color="#dcdcdc", linewidth=0.35, zorder=0)
    # Constant-Macro grid lines (horizontal, width follows trapezoid skew)
    for t in (0.25, 0.50, 0.75):
        mac = t * macro_max
        ax_inset.plot([m_lo + 0.5 * mac, m_hi + 0.5 * mac],
                      [h * t, h * t],
                      color="#dcdcdc", linewidth=0.35, zorder=0)

    # Bottom-edge: Meso below, Fine above (Fine = 1 - Meso at Macro=0)
    for m in meso_ticks:
        ax_inset.text(m, -0.040, f"{int(m*100)}", ha="center", va="top",
                       fontsize=4.6, color=DIM_COLORS["meso"])
        ax_inset.text(m, 0.018, f"{int((1-m)*100)}", ha="center", va="bottom",
                       fontsize=4.6, color=DIM_COLORS["fine"])

    # Macro ticks on right edge
    for t in (0.25, 0.5, 0.75):
        mac = t * macro_max
        x = m_hi + 0.5 * mac
        y = h * t
        ax_inset.text(x + 0.020, y, f"{mac*100:g}%",
                       ha="left", va="center", fontsize=4.4,
                       color=DIM_COLORS["macro"], rotation=-60)

    # Corner labels — adjusted to reflect the windowed range
    ax_inset.text(m_lo - 0.02, -0.04, f"F {int((1-m_lo)*100)}%",
                   ha="right", va="top",
                   color=DIM_COLORS["fine"], fontweight="bold", fontsize=5.0)
    ax_inset.text(m_hi + 0.02, -0.04, f"M {int(m_hi*100)}%",
                   ha="left", va="top",
                   color=DIM_COLORS["meso"], fontweight="bold", fontsize=5.0)
    ax_inset.text((m_lo + m_hi) / 2 + 0.25 * macro_max, h + 0.035,
                   f"Macro {int(macro_max*100)}%",
                   ha="center", va="bottom",
                   color=DIM_COLORS["macro"], fontweight="bold", fontsize=5.0)

    # xlim tracks the visible window with a margin for labels
    span = m_hi - m_lo
    margin = 0.20 * span
    ax_inset.set_xlim(m_lo - margin, m_hi + 0.5 * macro_max + margin)
    ax_inset.set_ylim(-0.18, h + 0.18)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")


# ---------------------------------------------------------------------------
# Data loading + per-group ternary computation
# ---------------------------------------------------------------------------

def load_records() -> pd.DataFrame:
    df = pd.read_csv(FEM_DIR / "records.csv")
    # FSE pool = any of the 9 labels fire OR uncategorized
    label_cols = ALL_LABELS
    df["_in_pool"] = (
        df[label_cols].any(axis=1) | df["uncategorized"].astype(bool)
    )
    return df[df["_in_pool"]].copy()


def per_step_normalized_ternary(sub: pd.DataFrame) -> Dict[str, float]:
    """Average barycentric coords for a group of FSEs."""
    has_any = sub[["fine_present", "meso_present", "macro_present"]].any(axis=1)
    s = sub[has_any].copy()
    if s.empty:
        return {"fine": 1 / 3, "meso": 1 / 3, "macro": 1 / 3, "n": 0}
    k = s[["fine_present", "meso_present", "macro_present"]].sum(axis=1)
    return {
        "fine":  float((s["fine_present"]  / k).mean()),
        "meso":  float((s["meso_present"]  / k).mean()),
        "macro": float((s["macro_present"] / k).mean()),
        "n": int(len(s)),
    }


def per_group_ternary(df: pd.DataFrame, group_col: str,
                      group_order: List) -> Dict[str, Dict[str, float]]:
    out = {}
    for g in group_order:
        sub = df[df[group_col] == g]
        if sub.empty:
            out[g] = {"fine": 1 / 3, "meso": 1 / 3, "macro": 1 / 3, "n": 0}
        else:
            out[g] = per_step_normalized_ternary(sub)
    return out


def overall_label_shares(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in ALL_LABELS:
        k = int(df[c].sum())
        rows.append({
            "label": c,
            "display": LABEL_DISPLAY[c],
            "dim": LABEL_DIM[c],
            "n": k,
            "share": (k / n) if n else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel (a) — Ternary scatter with size / model / difficulty markers
# ---------------------------------------------------------------------------

def _size_marker_area(sz: int) -> float:
    """Linear scaling: s3 small, s30 large."""
    return 14 + (sz - SIZE_ORDER[0]) * 4.2


def _viz_zorder(area: float) -> float:
    z = 6.0 - area / 50.0
    return max(4.0, min(6.0, z))


def _plot_conditions(ax, by_size, by_model, by_diff,
                      macro_max: float = 1.0,
                      y_height: float | None = None,
                      marker_scale: float = 1.0,
                      with_size_trend: bool = True) -> None:
    """Common scatter code. `y_height` controls vertical extent of the
    projection (see tern_to_xy); pass a smaller value for a squashed inset."""
    size_xy = [tern_to_xy(by_size[sz]["fine"], by_size[sz]["meso"],
                          by_size[sz]["macro"],
                          macro_max=macro_max, y_height=y_height)
               for sz in SIZE_ORDER]
    if with_size_trend:
        sxs, sys = zip(*size_xy)
        ax.plot(sxs, sys, color="#888", linewidth=0.55,
                linestyle=(0, (3, 2)), zorder=3, alpha=0.65)

    for sz, (x, y) in zip(SIZE_ORDER, size_xy):
        area = _size_marker_area(sz) * marker_scale
        ax.scatter([x], [y], marker="o", s=area,
                   facecolor="#cfcfcf", edgecolor="#555", linewidth=0.5,
                   zorder=_viz_zorder(area), alpha=0.9)

    for m in MODEL_ORDER:
        s = by_model[m]
        x, y = tern_to_xy(s["fine"], s["meso"], s["macro"],
                          macro_max=macro_max, y_height=y_height)
        ax.scatter([x], [y], marker="*", s=190 * marker_scale,
                   facecolor=MODEL_COLORS[m],
                   edgecolor="white", linewidth=0.4, alpha=0.95,
                   zorder=_viz_zorder(190) + 0.5)

    for d in DIFF_ORDER:
        s = by_diff[d]
        x, y = tern_to_xy(s["fine"], s["meso"], s["macro"],
                          macro_max=macro_max, y_height=y_height)
        ax.scatter([x], [y], marker="s", s=36 * marker_scale,
                   facecolor=DIFF_COLORS[d],
                   edgecolor="white", linewidth=0.4, alpha=0.95,
                   zorder=_viz_zorder(36) + 0.5)


def panel_a_ternary(ax, by_size, by_model, by_diff, ax_inset=None):
    draw_ternary_axes(ax)
    _plot_conditions(ax, by_size, by_model, by_diff,
                     macro_max=1.0, marker_scale=1.0, with_size_trend=True)

    h = np.sqrt(3) / 2
    if ax_inset is not None:
        draw_inset_trapezoid(ax_inset, macro_max=INSET_MACRO_MAX,
                              meso_range=INSET_MESO_RANGE,
                              y_height=INSET_Y_HEIGHT)
        _plot_conditions(ax_inset, by_size, by_model, by_diff,
                         macro_max=INSET_MACRO_MAX,
                         y_height=INSET_Y_HEIGHT,
                         marker_scale=0.65, with_size_trend=True)

        # Thin grey rubber band from the main-triangle data cluster to the
        # inset's right edge (the inset sits to the left of the triangle).
        from matplotlib.patches import ConnectionPatch
        con = ConnectionPatch(
            xyA=(0.05, 0.04), coordsA=ax.transData,
            xyB=(INSET_MESO_RANGE[1] + 0.06, INSET_Y_HEIGHT * 0.5),
            coordsB=ax_inset.transData,
            color="#bbb", linewidth=0.5, linestyle=(0, (1.5, 1.5)),
            zorder=1,
        )
        ax.add_artist(con)

    ax.set_title("Multi-label first-error composition", pad=4)


def draw_panel_a_legend(ax_leg) -> None:
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    y = 0.97
    row_h = 0.054
    section_gap = 0.025
    marker_x = 0.10
    text_x = 0.28

    ax_leg.text(0, y, "Maze size", fontsize=6.6, fontweight="bold",
                color="#222", va="top")
    y -= row_h
    for sz in SIZE_ORDER:
        ax_leg.scatter([marker_x], [y], marker="o",
                       s=_size_marker_area(sz),
                       facecolor="#cfcfcf", edgecolor="#555", linewidth=0.5,
                       alpha=0.85)
        ax_leg.text(text_x, y, f"s{sz}  ({sz}x{sz})",
                    fontsize=5.8, va="center", color="#444")
        y -= row_h

    y -= section_gap
    ax_leg.text(0, y, "Model", fontsize=6.6, fontweight="bold",
                color="#222", va="top")
    y -= row_h
    for m in MODEL_ORDER:
        ax_leg.scatter([marker_x], [y], marker="*", s=110,
                       facecolor=MODEL_COLORS[m],
                       edgecolor="none", linewidth=0, alpha=0.85)
        ax_leg.text(text_x, y, MODEL_LABELS[m], fontsize=5.8, va="center",
                    color=MODEL_COLORS[m], fontweight="bold")
        y -= row_h

    y -= section_gap
    ax_leg.text(0, y, "Difficulty", fontsize=6.6, fontweight="bold",
                color="#222", va="top")
    y -= row_h
    for d in DIFF_ORDER:
        ax_leg.scatter([marker_x], [y], marker="s", s=30,
                       facecolor=DIFF_COLORS[d],
                       edgecolor="none", linewidth=0, alpha=0.85)
        ax_leg.text(text_x, y, d.capitalize(), fontsize=5.8, va="center",
                    color=DIFF_COLORS[d], fontweight="bold")
        y -= row_h


# ---------------------------------------------------------------------------
# Panel (b) — 8-label multi-hot bar chart
# ---------------------------------------------------------------------------

def panel_b_multilabel(ax, df_shares: pd.DataFrame, n_fse: int) -> None:
    """Horizontal bars, grouped by dimension top→bottom, share = P(label fires).

    `fine.empty_or_unparsed_path` is intentionally excluded: it tracks a
    pre-step failure (LLM produced no usable path at all) rather than a
    geometric step-level error, so it doesn't fit visually alongside the
    other 8 labels.
    """
    df_shares = df_shares[df_shares["label"] != "fine.empty_or_unparsed_path"]
    rows = []
    for dim in ["fine", "meso", "macro"]:
        sub = df_shares[df_shares["dim"] == dim].sort_values("share",
                                                              ascending=False)
        rows.append(sub)
    ordered = pd.concat(rows, ignore_index=True)

    y = np.arange(len(ordered))[::-1]
    colors = [LABEL_COLORS[k] for k in ordered["label"]]

    ax.barh(y, ordered["share"], color=colors, edgecolor="white",
            linewidth=0.4, height=0.78)

    for i, (sh, nv) in enumerate(zip(ordered["share"], ordered["n"])):
        ax.text(sh + 0.006, y[i], f"{sh*100:.1f}%  (n={int(nv)})",
                va="center", ha="left", fontsize=5.4, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels([LABEL_DISPLAY[k] for k in ordered["label"]])
    ax.set_xlim(0, ordered["share"].max() * 1.20)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    ax.set_xticklabels(["0", "10", "20", "30", "40"])
    ax.set_xlabel("Share of FSE pool (multi-hot, sum > 100%)")
    ax.set_title("Per-label share", pad=2)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)

    _draw_dim_brackets(ax, ordered)


def _draw_dim_brackets(ax, ordered: pd.DataFrame) -> None:
    """Draw a thin vertical color bracket on the left of bars grouped by dim."""
    y_positions = np.arange(len(ordered))[::-1]
    by_dim_y: Dict[str, List[float]] = {}
    for y, dim in zip(y_positions, ordered["dim"]):
        by_dim_y.setdefault(dim, []).append(y)

    # Place bracket just outside the y-axis tick labels area.
    # Use axes-fraction transform for x, data coords for y.
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    bracket_x = -0.21   # axes-fraction. Closer to y-axis → snug to label text
    label_x = -0.24

    for dim, ys in by_dim_y.items():
        y_top = max(ys) + 0.42
        y_bot = min(ys) - 0.42
        ax.plot([bracket_x, bracket_x], [y_top, y_bot],
                transform=tr, color=DIM_COLORS[dim], linewidth=2.2,
                solid_capstyle="round", clip_on=False)
        ax.text(label_x, (y_top + y_bot) / 2, DIM_LABELS[dim],
                transform=tr, color=DIM_COLORS[dim],
                fontsize=6.8, fontweight="bold",
                ha="right", va="center", rotation=90, clip_on=False)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_figure(df_pool, by_size, by_model, by_diff, df_shares):
    fig_w = 150 / MM_PER_INCH
    fig_h = 175 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.20, 1.20],
        hspace=0.26,
        left=0.16, right=0.93,
        top=0.96, bottom=0.06,
    )
    # Panel a: three sub-columns [inset | triangle | legend].
    gs_a = gs[0, 0].subgridspec(
        1, 3,
        width_ratios=PANEL_A_WIDTH_RATIOS,
        wspace=0.10,
    )
    ax_a_inset = fig.add_subplot(gs_a[0, 0])
    ax_a_tri   = fig.add_subplot(gs_a[0, 1])
    ax_a_leg   = fig.add_subplot(gs_a[0, 2])
    ax_b = fig.add_subplot(gs[1, 0])

    panel_a_ternary(ax_a_tri, by_size, by_model, by_diff,
                     ax_inset=ax_a_inset)
    draw_panel_a_legend(ax_a_leg)
    panel_b_multilabel(ax_b, df_shares, n_fse=len(df_pool))

    # Panel labels "a" / "b" at a SHARED figure-x so they line up vertically.
    # Both pinned just above their respective gridspec row top.
    label_x = 0.035  # figure-x; sits left of any panel content
    pa_top = ax_a_inset.get_position().y1
    pb_top = ax_b.get_position().y1
    fig.text(label_x, pa_top + 0.012, "a",
             fontsize=9, fontweight="bold", va="bottom", ha="left")
    fig.text(label_x, pb_top + 0.012, "b",
             fontsize=9, fontweight="bold", va="bottom", ha="left")

    return fig


def build_panel_a_only(df_pool, by_size, by_model, by_diff):
    """Standalone Panel-a figure: ternary + zoom inset + legend.
    Same width ratios as the combined figure; sized 150 mm × 80 mm.
    """
    fig_w = 150 / MM_PER_INCH
    fig_h = 80 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=PANEL_A_WIDTH_RATIOS,
        wspace=0.10,
        left=0.08, right=0.965,
        top=0.92, bottom=0.07,
    )
    ax_inset = fig.add_subplot(gs[0, 0])
    ax_tri   = fig.add_subplot(gs[0, 1])
    ax_leg   = fig.add_subplot(gs[0, 2])

    panel_a_ternary(ax_tri, by_size, by_model, by_diff, ax_inset=ax_inset)
    draw_panel_a_legend(ax_leg)

    # Panel label "a" at the figure's top-left.
    fig.text(0.015, 0.965, "a",
             fontsize=9, fontweight="bold", va="top", ha="left")
    return fig


def build_panel_b_only(df_pool, df_shares):
    """Standalone Panel-b figure: 8-label multi-hot bar chart. 150 mm × 80 mm."""
    fig_w = 150 / MM_PER_INCH
    fig_h = 80 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=1, ncols=1,
        left=0.22, right=0.93,
        top=0.90, bottom=0.13,
    )
    ax = fig.add_subplot(gs[0, 0])
    panel_b_multilabel(ax, df_shares, n_fse=len(df_pool))

    # Panel label "b" at the figure's top-left.
    fig.text(0.015, 0.965, "b",
             fontsize=9, fontweight="bold", va="top", ha="left")
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
                        default="paper/figures/fig4_first_error_multilabel")
    args = parser.parse_args()

    setup_style()
    df_pool = load_records()
    print(f"FSE pool: {len(df_pool):,}")

    by_size  = per_group_ternary(df_pool, "effective_size", SIZE_ORDER)
    by_model = per_group_ternary(df_pool, "model", MODEL_ORDER)
    by_diff  = per_group_ternary(df_pool, "difficulty", DIFF_ORDER)
    df_shares = overall_label_shares(df_pool)

    print("\nPer-group ternary centroids (per-step-normalized):")
    for label, table in [("size", by_size), ("model", by_model),
                         ("difficulty", by_diff)]:
        print(f"  by {label}:")
        for g, s in table.items():
            chunk = " ".join(f"{d}={s[d]:.2f}" for d in DIMS)
            print(f"    {g:>18}  ({chunk})  n={s['n']}")

    print("\nLabel marginal shares:")
    for _, r in df_shares.iterrows():
        print(f"  {r['label']:30s}  share={r['share']:.4f}  n={int(r['n'])}")

    fig = build_figure(df_pool, by_size, by_model, by_diff, df_shares)
    out = PROJECT_ROOT / args.out
    save_pub(fig, out)
    print(f"\nSaved {out}.{{svg,pdf,png,tiff}}")
    plt.close(fig)

    fig_a = build_panel_a_only(df_pool, by_size, by_model, by_diff)
    out_a = PROJECT_ROOT / (args.out + "_panel_a")
    save_pub(fig_a, out_a)
    print(f"Saved {out_a}.{{svg,pdf,png,tiff}}")
    plt.close(fig_a)

    fig_b = build_panel_b_only(df_pool, df_shares)
    out_b = PROJECT_ROOT / (args.out + "_panel_b")
    save_pub(fig_b, out_b)
    print(f"Saved {out_b}.{{svg,pdf,png,tiff}}")
    plt.close(fig_b)


if __name__ == "__main__":
    main()
