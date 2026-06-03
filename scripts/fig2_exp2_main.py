"""Nature-style Figure 2: Exp 2 main results (RQ2).

Claim: Scaling failure is driven by aggregation, not local perception.
One-shot SR collapses to 0 by 10x10, while isolated diagnostics preserve
allocentric orientation (Macro) but reveal weak metric grounding (Fine)
and fragile topological simulation (Meso).

Panels (183 mm x 85 mm):
  (a) One-shot SR vs maze size, three models, Bernoulli SEM band.
  (b) Independent diagnostic accuracy vs maze size, four dimensions,
      averaged across models with SEM band.
  (c) Composition vs isolation gap: at the collapse scale, oneshot SR is
      contrasted with each dimension's isolated accuracy (bars + SEM).

Usage:
    python -m analysis.fig2_exp2_main
    python analysis/fig2_exp2_main.py
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

# ---------------------------------------------------------------------------
# Canonical run directories (override on CLI if needed)
# ---------------------------------------------------------------------------
ONESHOT_DIR = PROJECT_ROOT / "exp_output" / "exp2_onehot"
INDEP_DIR = PROJECT_ROOT / "exp_output" / "exp2_independent" / "medium"

SIZES = [3, 5, 7, 10, 15, 20, 30]
MODELS_ORDER = ["gpt-4o", "deepseek-v3", "llama-3.3-70b"]
MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "deepseek-v3": "DeepSeek-V3",
    "llama-3.3-70b": "Llama-3.3-70B",
}
MODEL_COLORS = {
    "gpt-4o":        "#2F5F8B",
    "deepseek-v3":   "#C2725A",
    "llama-3.3-70b": "#6F9CC2",
}
DIMENSIONS = ["fine", "meso", "macro", "nonspatial"]
DIM_LABELS = {
    "fine":       "Fine",
    "meso":       "Meso",
    "macro":      "Macro",
    "nonspatial": "NonSpatial",
}
DIM_COLORS = {
    "fine":       "#4F8FCB",
    "meso":       "#E2A24A",
    "macro":      "#7A5BA0",
    "nonspatial": "#6E6E6E",
}
# Coupling colour for the Meso × Macro coupled probe (junction_branch).
# Kept in sync with Supp Fig 2 (figs2_coupling_ladder) — see captions.md
# "Cross-figure conventions: Coupling colour".
COUPLING_COLOR = "#9C3E3E"
# Per-dimension chance baseline. Meso (pure) is dead_end_recognition yes/no -> 0.50;
# the other dimensions are 4-choice MCQ -> 0.25.
DIM_CHANCE = {
    "fine":       0.25,
    "meso":       0.50,
    "macro":      0.25,
    "nonspatial": 0.25,
}
# Subset filter: when computing the canonical accuracy for a dimension, restrict to
# these question subtypes. None = use all (pooled) for that dimension.
DIM_QTYPE_FILTER: Dict[str, List[str]] = {
    "fine":       [],  # only one subtype (passable_directions)
    "meso":       ["dead_end_recognition"],  # exclude junction_branch (Meso x Macro coupling)
    "macro":      [],
    "nonspatial": [],
}
COLLAPSE_SCALE = 10  # scale used for the panel-c isolation-vs-composition contrast


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
# Stats
# ---------------------------------------------------------------------------

def bernoulli_sem(p: float, n: int, z: float = 1.96) -> float:
    """Wilson 95% half-width for a Bernoulli proportion (symmetric approx).

    Replaces the +/- 1 SEM half-width: the SEM degenerates to 0 at p=0 and
    p=1 (the saturated cells that dominate this benchmark), whereas the Wilson
    interval stays non-degenerate and gives 95% coverage. We return the
    symmetric half-width so downstream symmetric error bars are unchanged in
    shape. Name kept for call-site compatibility.
    """
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    return (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def _size_to_int(s: str) -> int:
    return int(s.lstrip("s"))


def load_oneshot() -> Dict[Tuple[str, int], Tuple[float, float, int]]:
    """{(model, size): (SR, SEM, n_total)} averaged over difficulty."""
    summary = json.load(open(ONESHOT_DIR / "summary.json"))
    by_ms: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for v in summary.values():
        m = v["model"]
        sz = _size_to_int(v["effective_size"])
        if m in MODELS_ORDER and sz in SIZES:
            by_ms[(m, sz)].append(v)

    out = {}
    for (m, sz), items in by_ms.items():
        n_total = sum(x["n_trials"] for x in items)
        sr = (sum(x["overall"]["SR"] * x["n_trials"] for x in items) / n_total
              if n_total else 0.0)
        sem = bernoulli_sem(sr, n_total)
        out[(m, sz)] = (sr, sem, n_total)
    return out


def _filtered_acc(group: dict, dim: str) -> Tuple[float, int]:
    """Compute (accuracy, n) for a (model, dim, size) group, restricted to the
    canonical question subtype(s) for that dimension.

    Returns the dimension's own pooled value when no subtype filter is set.
    """
    qtypes = DIM_QTYPE_FILTER.get(dim, [])
    by_qt = group.get("by_question_type", {})
    if qtypes and by_qt:
        n_total = sum(by_qt[qt]["n"] for qt in qtypes if qt in by_qt)
        if n_total > 0:
            n_correct = sum(by_qt[qt]["n_correct"] for qt in qtypes if qt in by_qt)
            return n_correct / n_total, n_total
    return group["accuracy"], group["n_trials"]


def load_independent_by_dim() -> Dict[Tuple[str, int], Tuple[float, float, int]]:
    """{(dim, size): (accuracy, SEM, n_total)} averaged across models.

    For Meso, only `dead_end_recognition` (pure-Meso probe) is included;
    `junction_branch` is a Meso x Macro coupled probe and is plotted separately.
    """
    summary = json.load(open(INDEP_DIR / "summary.json"))
    by_ds: Dict[Tuple[str, int], List[Tuple[float, int]]] = defaultdict(list)
    for v in summary.values():
        d = v["dimension"]
        sz = _size_to_int(v["effective_size"])
        if d in DIMENSIONS and sz in SIZES:
            by_ds[(d, sz)].append(_filtered_acc(v, d))

    out = {}
    for (d, sz), items in by_ds.items():
        n_total = sum(n for _, n in items)
        acc = (sum(a * n for a, n in items) / n_total) if n_total else 0.0
        out[(d, sz)] = (acc, bernoulli_sem(acc, n_total), n_total)
    return out


def load_independent_by_model_dim() -> Dict[Tuple[str, str, int], Tuple[float, float, int]]:
    """{(model, dim, size): (accuracy, SEM, n)} -- for the panel-c bars."""
    summary = json.load(open(INDEP_DIR / "summary.json"))
    out = {}
    for v in summary.values():
        m = v["model"]; d = v["dimension"]; sz = _size_to_int(v["effective_size"])
        if m not in MODELS_ORDER or d not in DIMENSIONS or sz not in SIZES:
            continue
        acc, n = _filtered_acc(v, d)
        out[(m, d, sz)] = (acc, bernoulli_sem(acc, n), n)
    return out


def load_meso_coupled() -> Dict[int, Tuple[float, float, int]]:
    """junction_branch (Meso x Macro coupled) accuracy by size, avg across models."""
    summary = json.load(open(INDEP_DIR / "summary.json"))
    by_sz: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for v in summary.values():
        if v["dimension"] != "meso":
            continue
        sz = _size_to_int(v["effective_size"])
        if sz not in SIZES:
            continue
        qt = v.get("by_question_type", {}).get("junction_branch")
        if qt and qt["n"] > 0:
            by_sz[sz].append((qt["n_correct"] / qt["n"], qt["n"]))
    out = {}
    for sz, items in by_sz.items():
        n = sum(n for _, n in items)
        acc = sum(a * n for a, n in items) / n if n else 0.0
        out[sz] = (acc, bernoulli_sem(acc, n), n)
    return out


def load_meso_coupled_by_model() -> Dict[Tuple[str, int],
                                           Tuple[float, float, int]]:
    """{(model, size): (acc, SEM, n)} for junction_branch — Meso x Macro
    coupled probe, used by panel c."""
    summary = json.load(open(INDEP_DIR / "summary.json"))
    out: Dict[Tuple[str, int], Tuple[float, float, int]] = {}
    for v in summary.values():
        if v["dimension"] != "meso":
            continue
        m = v["model"]
        sz = _size_to_int(v["effective_size"])
        if m not in MODELS_ORDER or sz not in SIZES:
            continue
        qt = v.get("by_question_type", {}).get("junction_branch")
        if not qt or qt["n"] == 0:
            continue
        acc = qt["n_correct"] / qt["n"]
        out[(m, sz)] = (acc, bernoulli_sem(acc, qt["n"]), qt["n"])
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _draw_line_with_sem(ax, x, y, ye, color, label, marker="o"):
    ax.plot(x, y, marker=marker, color=color, label=label,
            linewidth=1.1, markersize=3.0, zorder=3)
    ax.fill_between(x, np.array(y) - np.array(ye), np.array(y) + np.array(ye),
                    color=color, alpha=0.18, linewidth=0, zorder=2)


def panel_a_oneshot(ax, oneshot):
    """SR vs maze size, 3 models on the same axes."""
    for m in MODELS_ORDER:
        ys, yes = [], []
        for sz in SIZES:
            sr, sem, n = oneshot.get((m, sz), (0.0, 0.0, 0))
            ys.append(sr); yes.append(sem)
        _draw_line_with_sem(ax, SIZES, ys, yes, MODEL_COLORS[m], MODEL_LABELS[m])

    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] - 1, SIZES[-1] + 1)
    ax.minorticks_off()
    ax.set_xlabel("Maze effective size")
    ax.set_ylabel("One-shot success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("One-shot navigation", pad=2)
    ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.2)
    ax.text(-0.16, 1.06, "a", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_b_dimensions(ax, indep, meso_coupled):
    """Accuracy vs maze size, four pure dimensions + coupled Meso x Macro line."""
    for d in DIMENSIONS:
        ys, yes = [], []
        for sz in SIZES:
            row = indep.get((d, sz), (0.0, 0.0, 0))
            ys.append(row[0]); yes.append(row[1])
        _draw_line_with_sem(ax, SIZES, ys, yes, DIM_COLORS[d], DIM_LABELS[d])

    # Coupled Meso x Macro probe (junction_branch): dashed Meso-colored line
    ys, yes = [], []
    for sz in SIZES:
        row = meso_coupled.get(sz, (0.0, 0.0, 0))
        ys.append(row[0]); yes.append(row[1])
    ax.plot(SIZES, ys, marker="s", markersize=2.6, linewidth=1.0,
            linestyle=(0, (3, 2)),
            color=COUPLING_COLOR, label="Meso x Macro",
            zorder=3)
    ax.fill_between(SIZES, np.array(ys) - np.array(yes),
                    np.array(ys) + np.array(yes),
                    color=COUPLING_COLOR, alpha=0.10, linewidth=0, zorder=2)

    # Chance-level references: 4-choice (0.25) for Fine/Macro/NS/Meso x Macro,
    # 2-choice (0.50) for pure Meso (yes/no dead-end recognition)
    chance_bbox = dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85)
    ax.axhline(0.25, color="#999", linewidth=0.5, linestyle=":", zorder=1)
    ax.text(22, 0.25, "4-choice chance", fontsize=5.4, color="#777",
            ha="center", va="center", bbox=chance_bbox, zorder=4)
    ax.axhline(0.50, color="#999", linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
    # Anchor the Meso (yes/no) chance label under the line at the left edge and
    # point an arrow up to the dashed line so it does not overlap data curves.
    ax.annotate(
        "Meso (yes/no) chance",
        xy=(2.4, 0.50), xytext=(2.4, 0.31),
        fontsize=5.2, color="#777",
        ha="left", va="top",
        bbox=chance_bbox, zorder=4,
        arrowprops=dict(arrowstyle="->", color="#777",
                        lw=0.5, shrinkA=0, shrinkB=2),
    )

    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] - 1, SIZES[-1] + 1)
    ax.minorticks_off()
    ax.set_xlabel("Maze effective size")
    ax.set_ylabel("Isolated MCQ accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("Isolated cognitive dimensions", pad=2)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.2, ncol=2,
              columnspacing=0.9, fontsize=5.6)
    ax.text(-0.16, 1.06, "b", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_c_gap(ax, oneshot, indep_md, meso_coupled_md,
                  scale: int = COLLAPSE_SCALE):
    """Bars at the collapse scale: oneshot SR vs each dimension accuracy.

    Grouped by model (3 model groups) x bars per
    (SR, Fine, Meso, Macro, Meso×Macro, NonSpatial). The Meso×Macro bar uses
    hatched fill in the Meso color to mark it as a coupled probe (matches the
    dashed Meso×Macro curve in panel b).
    """
    # Insert Meso×Macro between Macro and NonSpatial.
    metrics       = ["SR", "fine", "meso", "macro", "meso_macro", "nonspatial"]
    metric_labels = ["One-shot SR", "Fine", "Meso", "Macro",
                     "Meso×Macro", "NonSpatial"]
    metric_colors = ["#444",
                      DIM_COLORS["fine"],
                      DIM_COLORS["meso"],
                      DIM_COLORS["macro"],
                      COUPLING_COLOR,        # Meso × Macro coupling colour
                      DIM_COLORS["nonspatial"]]

    n_models = len(MODELS_ORDER)
    n_bars = len(metrics)
    bar_w = 0.80 / n_bars
    x = np.arange(n_models)

    for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
        means, errs = [], []
        for m in MODELS_ORDER:
            if metric == "SR":
                v = oneshot.get((m, scale), (0.0, 0.0, 0))
            elif metric == "meso_macro":
                v = meso_coupled_md.get((m, scale), (0.0, 0.0, 0))
            else:
                v = indep_md.get((m, metric, scale), (0.0, 0.0, 0))
            means.append(v[0]); errs.append(v[1])
        offsets = x - 0.40 + (j + 0.5) * bar_w

        # One-shot SR and Meso×Macro both use the white-"////"-on-solid hatch
        # style (fig3 panel c convention): SR marks the "composed task" bar,
        # Meso×Macro marks the coupled Meso variant.
        hatched = metric in ("SR", "meso_macro")
        ax.bar(offsets, means, bar_w * 0.92,
                color=color, edgecolor="white", linewidth=0.4,
                hatch="////" if hatched else None,
                label=metric_labels[j])
        ax.errorbar(offsets, means, yerr=errs, fmt="none",
                    ecolor="#333", elinewidth=0.5, capsize=1.2, capthick=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS_ORDER],
                       rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Composition vs isolation @ {scale}x{scale}", pad=2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
              handlelength=0.9, handleheight=0.8, ncol=2,
              columnspacing=0.8, fontsize=5.6, borderaxespad=0.3)
    ax.text(-0.20, 1.06, "c", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_figure(oneshot, indep, indep_md, meso_coupled, meso_coupled_md):
    fig_w = 183 / MM_PER_INCH
    fig_h = 85 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 1.0, 1.05],
        wspace=0.34,
        left=0.06, right=0.985,
        top=0.88, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_oneshot(ax_a, oneshot)
    panel_b_dimensions(ax_b, indep, meso_coupled)
    panel_c_gap(ax_c, oneshot, indep_md, meso_coupled_md, COLLAPSE_SCALE)

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
                        default="paper/figures/fig2_exp2_main")
    parser.add_argument("--collapse-scale", type=int, default=COLLAPSE_SCALE)
    args = parser.parse_args()

    setup_style()
    oneshot = load_oneshot()
    indep = load_independent_by_dim()
    indep_md = load_independent_by_model_dim()
    meso_coupled = load_meso_coupled()
    meso_coupled_md = load_meso_coupled_by_model()

    print("Oneshot summary (model, size, SR, n):")
    for (m, sz), (sr, sem, n) in sorted(oneshot.items()):
        print(f"  {m:<16} s{sz:<3} SR={sr:.3f}  SEM={sem:.3f}  n={n}")

    print("Independent dim summary (dim, size, acc, n):")
    for (d, sz), (acc, sem, n) in sorted(indep.items()):
        print(f"  {d:<10} s{sz:<3} acc={acc:.3f}  SEM={sem:.3f}  n={n}")

    fig = build_figure(oneshot, indep, indep_md, meso_coupled,
                        meso_coupled_md)
    out = PROJECT_ROOT / args.out
    save_pub(fig, out)
    print(f"Saved {out}.{{svg,pdf,png,tiff}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
