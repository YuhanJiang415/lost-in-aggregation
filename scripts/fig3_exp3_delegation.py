"""Nature-style Figure 3: Exp 3 delegation ladder (RQ3).

Claim: Topology-Aided delegation lifts SR by 30-92 pp at 7x7-15x15
across both models (vs the One-shot baseline), while Topology-Blind
stays within +-10 pp of the baseline -- so the lift comes from the explicit
cell-type framing in the prompt, not from junction-level decision granularity
per se. By 20x20 the Aided gain narrows (+30 pp GPT-4o, 0 pp DeepSeek-V3,
n=10) and by 30x30 essentially no signal remains in either variant.

Panels (183 mm x 75 mm):
  (a) GPT-4o: SR vs maze size for one-shot, topology-blind (blind), topology-aided (aided).
  (b) DeepSeek-V3: same.
  (c) Delta SR bars: topology-aided minus one-shot baseline, per model per size.

Data points with n<50 are drawn with hollow markers and dashed line segments
to mark them as exploratory. SEM bands extend across both full (n=50) and
exploratory cells for visual continuity.
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

# Canonical run directories
ONESHOT_DIR = PROJECT_ROOT / "exp_output" / "exp2_onehot"
STEP_DIR    = PROJECT_ROOT / "exp_output" / "exp3_junction_blind"
JCT_DIR     = PROJECT_ROOT / "exp_output" / "exp3_junction_aided"

SIZES = [7, 10, 15, 20, 30]   # include s30 to make the SR=0 plateau visible
MODELS = ["gpt-4o", "deepseek-v3"]
MODEL_LABELS = {"gpt-4o": "GPT-4o", "deepseek-v3": "DeepSeek-V3"}

METHODS = ["one-shot", "topology-blind", "topology-aided"]
METHOD_LABELS = {
    "one-shot":       "One-shot",
    "topology-blind":   "Topology-blind Junction",
    "topology-aided": "Topology-aided Junction",
}
METHOD_COLORS = {
    # Method palette uses two hue families (teal + wine) NOT occupied by
    # the model / dimension palettes, so Topology-aided cannot be confused
    # with DeepSeek-V3 (#C2725A) and Topology-blind cannot be confused with
    # Fine (#4F8FCB).
    "one-shot":       "#7E7E7E",   # neutral grey baseline
    "topology-blind":   "#3E8C8C",   # teal — passive (Topology-blind)
    "topology-aided": "#9C4A8A",   # wine — signal (Topology-aided)
}
METHOD_MARKERS = {
    "one-shot":       "o",
    "topology-blind":   "s",
    "topology-aided": "D",
}

N_FULL = 50  # if n_total < N_FULL, treat the point as exploratory


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


def _load_summary_method(path: Path, method_tag: str,
                         medium_only: bool = False
                        ) -> Dict[Tuple[str, int], Tuple[float, float, int]]:
    """{(model, size): (SR, SEM, n)} for one method."""
    summary = json.load(open(path))
    agg: Dict[Tuple[str, int], list] = defaultdict(list)
    for v in summary.values():
        sz = _size_to_int(v["effective_size"])
        m = v["model"]
        if m not in MODELS:
            continue
        if medium_only and v.get("difficulty") != "medium":
            continue
        n = v["n_trials"]
        sr = v["overall"]["SR"]
        agg[(m, sz)].append((sr, n))
    out = {}
    for (m, sz), items in agg.items():
        n_total = sum(n for _, n in items)
        sr = (sum(sr * n for sr, n in items) / n_total) if n_total else 0.0
        out[(m, sz)] = (sr, bernoulli_sem(sr, n_total), n_total)
    return out


def _augment_from_jsonl(out: Dict[Tuple[str, int], Tuple[float, float, int]],
                        run_dir: Path, sizes: List[int],
                        medium_only: bool = False) -> None:
    """Fill in (model, size) cells that the summary aggregator missed by
    counting raw trials in `results_s{size}.jsonl` directly.

    Idempotent: existing entries in `out` are preserved. When `medium_only`
    is set, only medium-difficulty trials are counted (the one-shot baseline
    is reported on medium, matching the delegation regimes).
    """
    for sz in sizes:
        path = run_dir / f"results_s{sz}.jsonl"
        if not path.exists():
            continue
        counts: Dict[str, list] = defaultdict(lambda: [0, 0])  # [n, k]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                m = r.get("model")
                if m not in MODELS:
                    continue
                if medium_only and r.get("difficulty") != "medium":
                    continue
                sr_flag = bool(r.get("metrics", {}).get("overall", {}).get("SR", False))
                counts[m][0] += 1
                counts[m][1] += 1 if sr_flag else 0
        for m, (n, k) in counts.items():
            if n == 0 or (m, sz) in out:
                continue
            p = k / n
            out[(m, sz)] = (p, bernoulli_sem(p, n), n)


def load_all() -> Dict[str, Dict[Tuple[str, int], Tuple[float, float, int]]]:
    out = {
        "one-shot":       _load_summary_method(
            ONESHOT_DIR / "summary.json", "one-shot", medium_only=True),
        "topology-blind":   _load_summary_method(
            STEP_DIR / "summary.json", "topology-blind"),
        "topology-aided": _load_summary_method(
            JCT_DIR / "summary.json", "topology-aided"),
    }
    # The summary aggregator skipped a few cells. For one-shot, its summary
    # only spans s3/s5/s7 (the Module-2A sweep), so backfill s10-s30 from the
    # medium-difficulty raw trials. For the junction regimes, backfill the
    # exploratory cells (notably s30) so the SR=0 plateau is visible.
    _augment_from_jsonl(out["one-shot"], ONESHOT_DIR, SIZES, medium_only=True)
    _augment_from_jsonl(out["topology-blind"], STEP_DIR, SIZES)
    _augment_from_jsonl(out["topology-aided"], JCT_DIR, SIZES)
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_method_line(ax, sizes, data_by_size, color, marker, label):
    """Plot SR vs size for one (method, model). Hollow marker + dashed segment
    where any endpoint has n < N_FULL."""
    full_xs, full_ys, full_yes = [], [], []
    explr_xs, explr_ys, explr_yes = [], [], []
    is_full = []
    for sz in sizes:
        if sz not in data_by_size:
            is_full.append(None)
            continue
        sr, sem, n = data_by_size[sz]
        full = (n >= N_FULL)
        is_full.append(full)
        if full:
            full_xs.append(sz); full_ys.append(sr); full_yes.append(sem)
        else:
            explr_xs.append(sz); explr_ys.append(sr); explr_yes.append(sem)

    # Wilson band — drawn ONLY over full-n (>= N_FULL) points. Exploratory
    # (n<50) points get hollow markers with NO band: they are shown for trend
    # only and carry no statistical inference, so a confidence ribbon there
    # would be misleading (and at n=10 is very wide).
    if full_xs:
        order = sorted(range(len(full_xs)), key=lambda i: full_xs[i])
        ord_x  = [full_xs[i]  for i in order]
        ord_y  = [full_ys[i]  for i in order]
        ord_se = [full_yes[i] for i in order]
        ax.fill_between(ord_x,
                        np.array(ord_y) - np.array(ord_se),
                        np.array(ord_y) + np.array(ord_se),
                        color=color, alpha=0.18, linewidth=0, zorder=2)

    # Full-n: solid line + filled markers (band already drawn above)
    if full_xs:
        ax.plot(full_xs, full_ys, marker=marker, linestyle="-",
                color=color, label=label, markerfacecolor=color,
                markeredgecolor=color, linewidth=1.1, zorder=3)

    # Exploratory (n<50): hollow markers, dashed connection to last full point
    if explr_xs:
        # Connect last full point to first exploratory if they are consecutive
        all_x = full_xs + explr_xs
        all_y = full_ys + explr_ys
        # Sort by x just in case
        order = sorted(range(len(all_x)), key=lambda i: all_x[i])
        ordered_x = [all_x[i] for i in order]
        ordered_y = [all_y[i] for i in order]
        # Find the segment that bridges full to exploratory and plot dashed
        for i in range(len(ordered_x) - 1):
            x0, x1 = ordered_x[i], ordered_x[i + 1]
            if x1 in explr_xs:
                ax.plot([x0, x1], [ordered_y[i], ordered_y[i + 1]],
                        linestyle="--", color=color, linewidth=0.9,
                        zorder=3)
        # Hollow markers for exploratory points
        ax.scatter(explr_xs, explr_ys, marker=marker, s=18,
                   facecolors="white", edgecolors=color, linewidths=0.9,
                   zorder=4)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_method_lines(ax, all_data, model, panel_letter, show_legend=False):
    for method in METHODS:
        data_by_size = {sz: all_data[method].get((model, sz))
                        for sz in SIZES
                        if all_data[method].get((model, sz)) is not None}
        _draw_method_line(
            ax, SIZES, data_by_size,
            METHOD_COLORS[method], METHOD_MARKERS[method],
            METHOD_LABELS[method],
        )
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] - 1, SIZES[-1] + 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Maze effective size")
    ax.set_ylabel("Success rate")
    ax.set_title(MODEL_LABELS[model], pad=2)
    ax.text(-0.18, 1.06, panel_letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")
    if show_legend:
        ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2,
                  handleheight=0.9)


def panel_delta_bars(ax, all_data):
    """Delta SR (topology-aided - one-shot), one bar group per size,
    sub-bars per model. Hatched bar where topology-aided n < 50."""
    n_sizes = len(SIZES)
    n_models = len(MODELS)
    bar_w = 0.72 / n_models
    x = np.arange(n_sizes)
    colors = {"gpt-4o": "#2F5F8B", "deepseek-v3": "#C2725A"}

    for j, model in enumerate(MODELS):
        offsets = x - 0.36 + (j + 0.5) * bar_w
        for i, sz in enumerate(SIZES):
            jo = all_data["topology-aided"].get((model, sz))
            os_ = all_data["one-shot"].get((model, sz))
            if jo is None or os_ is None:
                # mark "no data" with a small dash at y=0
                ax.text(offsets[i], 0.02, "n/a",
                        ha="center", va="bottom",
                        fontsize=4.6, color="#888",
                        rotation=90)
                continue
            d = jo[0] - os_[0]
            sem = math.sqrt(jo[1] ** 2 + os_[1] ** 2)
            bar = ax.bar(offsets[i], d, bar_w * 0.92,
                         color=colors[model], edgecolor="white",
                         linewidth=0.4,
                         label=MODEL_LABELS[model] if i == 0 else None)[0]
            # Error bar only for full-n cells; exploratory (n<50) bars are
            # hatched and shown for trend only, without statistical inference.
            if jo[2] >= N_FULL:
                ax.errorbar(offsets[i], d, yerr=sem, fmt="none",
                            ecolor="#333", elinewidth=0.5,
                            capsize=1.2, capthick=0.5)
            else:
                bar.set_hatch("////")
                bar.set_edgecolor("white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(-0.5, n_sizes - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Maze effective size")
    ax.set_ylabel("Δ Success rate\n(Junctions − One-shot)")
    ax.axhline(0, color="#444", linewidth=0.5, linestyle="-", zorder=1)
    ax.set_title("Junction-delegation gain", pad=2)
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2,
              handleheight=0.9)
    ax.text(-0.20, 1.06, "c", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")
    # Footnote about hatched bars (small-n)
    ax.text(0.99, -0.22,
            "Hatched: Topology Junction n<50 (exploratory)",
            transform=ax.transAxes, fontsize=5.4, color="#666",
            ha="right", va="top")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_figure(all_data):
    fig_w = 183 / MM_PER_INCH
    fig_h = 75 / MM_PER_INCH
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 1.0, 1.05],
        wspace=0.36,
        left=0.06, right=0.985,
        top=0.86, bottom=0.21,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_method_lines(ax_a, all_data, "gpt-4o", "a", show_legend=True)
    panel_method_lines(ax_b, all_data, "deepseek-v3", "b", show_legend=False)
    panel_delta_bars(ax_c, all_data)
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
                        default="paper/figures/fig3_exp3_delegation")
    args = parser.parse_args()

    setup_style()
    all_data = load_all()
    print("Method x model x size SR (n in parens):")
    for method in METHODS:
        for (m, sz), (sr, sem, n) in sorted(all_data[method].items()):
            print(f"  {method:<16} {m:<14} s{sz:<3} SR={sr:.3f}  SEM={sem:.3f}  n={n}")

    fig = build_figure(all_data)
    out = PROJECT_ROOT / args.out
    save_pub(fig, out)
    print(f"Saved {out}.{{svg,pdf,png,tiff}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
