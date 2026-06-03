"""Supplementary Figure 2: coupling ladder.

Claim: Each rung up the coupling ladder costs accuracy. Pure single-dimension
probes (Fine, Meso, Macro) stay well above chance across scales; a single
two-dimension coupling (Meso x Macro, junction_branch) already drops toward
its 4-choice chance floor; full-sequence one-shot navigation collapses to 0
by 10x10. Plotted left-to-right: pure_macro -> pure_meso -> pure_fine ->
meso x macro coupling -> oneshot SR.

Panels (130 mm x 150 mm, stacked vertically):
  (a) Scale-decay retention: each rung's accuracy at each maze size is
      normalized to its OWN value at 3x3, so every curve starts at 1.0 and
      the curve's slope directly visualizes the rate of decay with scale.
      One-shot collapses to 0 fastest; pure single-dimension probes retain
      60-86 % of their small-scale accuracy at 30x30.
  (b) Above-chance normalized score (acc - chance) / (1 - chance), 5 lines,
      single 0 baseline so the impact of each coupling rung is directly
      comparable across maze sizes.

Data:
  pure_fine               = independent.passable_directions (4-choice, c=0.25)
  pure_meso               = independent.dead_end_recognition (yes/no, c=0.50)
  pure_macro              = independent.midcourse_direction (4-choice, c=0.25)
  meso x macro coupling   = independent.junction_branch (4-choice, c=0.25)
  oneshot SR (full chain) = exp2_oneshot SR (medium only, c=0)
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
ONESHOT_SUMMARY = (PROJECT_ROOT / "exp_output" / "exp2_onehot"
                   / "summary.json")

SIZES = [3, 5, 7, 10, 15, 20, 30]

# Each "rung" of the ladder: (key, label, color, linestyle, chance level).
# Order: Fine -> Meso -> Macro (matching Fig 1/2 dimension order) followed by
# the Meso x Macro coupled probe and the full one-shot sequence.
RUNGS: List[Tuple[str, str, str, str, float]] = [
    ("pure_fine",   "Fine",          "#4F8FCB", "-",  0.25),
    ("pure_meso",   "Meso",          "#E2A24A", "-",  0.50),
    ("pure_macro",  "Macro",         "#7A5BA0", "-",  0.25),
    ("meso_macro",  "Meso × Macro",  "#9C3E3E", "--", 0.25),
    ("oneshot_sr",  "One-shot",      "#1F1F1F", "-",  0.00),
]

# Per-rung vertical offset (in points) for the end-of-line retention label
# in panel a. Resolves the ~60–63 % cluster (Fine / Macro / Meso × Macro).
_RETENTION_LABEL_DY: Dict[str, float] = {
    "pure_fine":   0.0,    # mid-cluster, sits at true position
    "pure_macro":  +9.0,   # 63 % -> nudge up
    "meso_macro":  -9.0,   # 60 % -> nudge down
    "pure_meso":   0.0,    # alone at 86 %
    "oneshot_sr":  0.0,    # alone at 0 %
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
        "legend.fontsize": 5.6,
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


def bernoulli_sem(p: float, n: int, z: float = 1.96) -> float:
    """Wilson 95% half-width for a Bernoulli proportion (symmetric approx).

    Replaces the +/- 1 SEM half-width for consistency with the other figures:
    the SEM degenerates to 0 at p=0/1 and only gives ~68% coverage, whereas
    Wilson stays non-degenerate at 95% coverage. The downstream linear
    propagation through the normalization / above-chance transforms applies
    unchanged (it just rescales this half-width). Name kept for call-site
    compatibility.
    """
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    return (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _qtype_acc_by_size(qtype_keys: List[str]
                       ) -> Dict[int, Tuple[float, float, int]]:
    """Aggregate accuracy across models for one or more by_question_type keys."""
    summary = json.load(open(INDEP_SUMMARY))
    agg: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for v in summary.values():
        sz = int(v["effective_size"].lstrip("s"))
        if sz not in SIZES:
            continue
        for qt in qtype_keys:
            stats = v.get("by_question_type", {}).get(qt)
            if not stats:
                continue
            nc, n = agg[sz]
            agg[sz] = (nc + stats["n_correct"], n + stats["n"])
    out: Dict[int, Tuple[float, float, int]] = {}
    for sz, (nc, n) in agg.items():
        if n == 0:
            continue
        p = nc / n
        out[sz] = (p, bernoulli_sem(p, n), n)
    return out


def _oneshot_sr_medium_by_size() -> Dict[int, Tuple[float, float, int]]:
    """SR for medium-difficulty oneshot, pooled across the 3 models.

    Read from the raw results_s*.jsonl rather than summary.json: the oneshot
    summary only aggregates the Module-2A sweep (s3/s5/s7), but the medium
    trials exist for all seven sizes in the JSONL, and the coupling ladder
    needs the full s3-s30 one-shot collapse curve.
    """
    onehot_dir = ONESHOT_SUMMARY.parent
    agg: Dict[int, list] = defaultdict(lambda: [0, 0])  # sz -> [n, k]
    for sz in SIZES:
        path = onehot_dir / f"results_s{sz}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("difficulty") != "medium":
                    continue
                agg[sz][0] += 1
                agg[sz][1] += 1 if bool(
                    r.get("metrics", {}).get("overall", {}).get("SR", False)
                ) else 0
    out: Dict[int, Tuple[float, float, int]] = {}
    for sz, (n, k) in agg.items():
        sr = (k / n) if n else 0.0
        out[sz] = (sr, bernoulli_sem(sr, n), n)
    return out


def load_all_rungs() -> Dict[str, Dict[int, Tuple[float, float, int]]]:
    return {
        "pure_macro":  _qtype_acc_by_size(["midcourse_direction"]),
        "pure_meso":   _qtype_acc_by_size(["dead_end_recognition"]),
        "pure_fine":   _qtype_acc_by_size(["passable_directions"]),
        "meso_macro":  _qtype_acc_by_size(["junction_branch"]),
        "oneshot_sr":  _oneshot_sr_medium_by_size(),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_rung(ax, data, color: str, style: str, label: str,
               normalized: bool = False, chance: float = 0.0):
    xs, ys, se = [], [], []
    for sz in SIZES:
        row = data.get(sz)
        if row is None:
            continue
        acc, sem, _n = row
        if normalized:
            denom = (1 - chance) if chance < 1 else 1e-9
            y_val = max(0.0, (acc - chance) / denom)
            # propagate SEM through the linear transform
            sem_val = sem / denom
        else:
            y_val = acc
            sem_val = sem
        xs.append(sz); ys.append(y_val); se.append(sem_val)
    ax.plot(xs, ys, marker="o", linestyle=style, color=color,
            linewidth=1.1, markersize=3.0, label=label, zorder=4)
    ax.fill_between(xs,
                    np.array(ys) - np.array(se),
                    np.array(ys) + np.array(se),
                    color=color, alpha=0.16, linewidth=0, zorder=2)


def panel_retention(ax, all_data):
    """Retention curves: accuracy at each maze size normalized to that
    rung's OWN accuracy at the smallest size (s3). All curves start at 1.0
    so the line's slope directly visualizes the decay rate with scale.
    The faster the drop, the less the rung tolerates increasing maze size.
    """
    baseline_size = SIZES[0]   # s3

    # Chance reference is not meaningful in normalized space; just a 0 line
    ax.axhline(0.0, color="#aaa", linewidth=0.4, linestyle=":", zorder=1)
    ax.axhline(1.0, color="#aaa", linewidth=0.4, linestyle=":", zorder=1)
    ax.text(SIZES[-1] - 0.4, 0.012, "0 = total loss",
            fontsize=4.8, color="#888", ha="right", va="bottom")
    ax.text(SIZES[-1] - 0.4, 1.012, "1 = full retention",
            fontsize=4.8, color="#888", ha="right", va="bottom")

    for key, label, color, style, _chance in RUNGS:
        d = all_data[key]
        base = d.get(baseline_size, (0.0,))[0]
        if base <= 0:
            continue
        xs, ys, se = [], [], []
        for sz in SIZES:
            row = d.get(sz)
            if row is None:
                continue
            acc, sem, _n = row
            xs.append(sz)
            ys.append(acc / base)
            se.append(sem / base)
        ax.plot(xs, ys, marker="o", linestyle=style, color=color,
                linewidth=1.1, markersize=3.0, label=label, zorder=4)
        ax.fill_between(xs,
                        np.array(ys) - np.array(se),
                        np.array(ys) + np.array(se),
                        color=color, alpha=0.16, linewidth=0, zorder=2)
        # End-of-line retention label -- offsets resolve the
        # Fine/Macro/Meso×Macro cluster (~60-63 %) so the three labels
        # stack vertically with leader lines instead of overlapping.
        if xs:
            final_x, final_y = xs[-1], ys[-1]
            y_off_pts = _RETENTION_LABEL_DY.get(key, 0)
            arrow = (dict(arrowstyle="-", color=color, lw=0.3,
                          shrinkA=0, shrinkB=1)
                     if y_off_pts != 0 else None)
            ax.annotate(f"{final_y*100:.0f}%",
                        (final_x, final_y),
                        xytext=(6, y_off_pts), textcoords="offset points",
                        fontsize=5.4, color=color, fontweight="bold",
                        va="center", ha="left",
                        arrowprops=arrow)

    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] - 1, SIZES[-1] + 3)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Maze effective size")
    ax.set_ylabel(f"Retention\n(acc / acc at {baseline_size}×{baseline_size})")
    ax.set_title("Scale-decay rate per rung", pad=2)
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.2,
              handleheight=0.9, fontsize=5.4)
    ax.text(-0.18, 1.06, "a", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_normalized(ax, all_data):
    # baseline at 0
    ax.axhline(0.0, color="#aaa", linewidth=0.5, linestyle=":", zorder=1)
    ax.text(SIZES[-1] - 0.4, 0.012, "chance baseline",
            fontsize=5.0, color="#777", ha="right", va="bottom")

    for key, label, color, style, chance in RUNGS:
        _plot_rung(ax, all_data[key], color, style, label,
                   normalized=True, chance=chance)

    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] - 1, SIZES[-1] + 1)
    ax.set_ylim(-0.05, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Maze effective size")
    ax.set_ylabel("Above-chance score\n(acc − chance) / (1 − chance)")
    ax.set_title("Coupling ladder (above-chance)", pad=2)
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2,
              handleheight=0.9, fontsize=5.0)
    ax.text(-0.20, 1.06, "b", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_figure(all_data):
    fig_w = 130 / MM_PER_INCH
    fig_h = 150 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        hspace=0.40,
        left=0.14, right=0.92,
        top=0.95, bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    panel_retention(ax_a, all_data)
    panel_normalized(ax_b, all_data)
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
                        default="paper/figures/figs2_coupling_ladder")
    args = parser.parse_args()

    setup_style()
    all_data = load_all_rungs()
    print("Rung values (raw acc, SEM, n) by maze size:")
    for key, label, _, _, _ in RUNGS:
        print(f"  {key}:")
        for sz in SIZES:
            row = all_data[key].get(sz)
            if row:
                print(f"    s{sz:<3} acc={row[0]:.3f}  SEM={row[1]:.3f}  n={row[2]}")

    fig = build_figure(all_data)
    out = PROJECT_ROOT / args.out
    save_pub(fig, out)
    print(f"Saved {out}.{{svg,pdf,png,tiff}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
