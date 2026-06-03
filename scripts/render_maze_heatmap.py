"""Render s5 (11x11 grid) mazes as junction-density heatmaps, one per
difficulty, in the visual style of `scripts/render_maze_preview.py`.

Density model (graph-aware, walls cannot leak heat):
  density(cell) = Σ_j exp(-d_graph(cell, j) / sigma)
where the sum runs over every junction cell j (degree >= 3) and
d_graph is the in-maze BFS distance from `cell` to `j`. Walls block
the BFS, so two parallel corridors separated by a wall do NOT
contaminate each other's density.

Colour rule (REVERSED from the earlier 'hot' version):
  Low density  → nearly white path cell  (easy / sparse local topology)
  High density → deep red path cell      (hard / clustered junctions)

The base maze styling matches `render_maze_preview.py`:
  - White path cells with thin grey grid borders (#dddddd)
  - Light-grey walls (#cfcfcf)
  - Blue/Red S/G dots with white letters
  - Optional dashed shortest-path overlay (kept for visual continuity)

Output: data/maze_previews/heatmap_s5_diff_comparison.{png,svg}
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle

REPO = Path(__file__).resolve().parent.parent
MAZE_FILE = REPO / "data" / "mazes" / "mazes_s5.json"
OUT_DIR = REPO / "data" / "maze_previews"

# Style — mirrors scripts/render_maze_preview.py
WALL_COLOR    = "#cfcfcf"
PATH_COLOR    = "#ffffff"           # density=0 path cells fall back to this
GRID_COLOR    = "#dddddd"
OPTIMAL_COLOR = "#aaaaaa"
START_COLOR   = "#3F70C8"
GOAL_COLOR    = "#E84B3A"

# Density colormap: light → dark. Density grows ⇒ colour darkens, matching
# the "easier = lighter, harder = darker" requirement.
DENSITY_CMAP = plt.cm.Reds

# Kernel width for the graph-distance density (in cell-steps).
DENSITY_SIGMA = 1.5


# ---------------------------------------------------------------------------
# Style + IO
# ---------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    })


def load_mazes() -> Dict[str, dict]:
    """Pick the first maze of each difficulty tier."""
    with open(MAZE_FILE) as f:
        all_mazes = json.load(f)
    picks: Dict[str, dict] = {}
    for diff in ("easy", "medium", "hard"):
        for m in all_mazes:
            if m.get("difficulty") == diff:
                picks[diff] = m
                break
    return picks


# ---------------------------------------------------------------------------
# Topology + density
# ---------------------------------------------------------------------------

def _degree(grid: np.ndarray) -> np.ndarray:
    """Per-cell count of passable orthogonal neighbours (path-cells only)."""
    rows, cols = grid.shape
    deg = np.zeros_like(grid, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    deg[r, c] += 1
    return deg


def junction_mask(grid: np.ndarray,
                  start: Tuple[int, int] | None = None) -> np.ndarray:
    """Path-cell mask of "junctions" — cells where the navigation agent
    has to choose.

    Rule (matches `experiments/exp3_cj_algoseg.py:538-540`):
      - any path cell with degree >= 3 is a junction;
      - additionally, the START cell is a junction whenever it has
        degree >= 2 (no came_from to exclude, so 2 forward options
        already requires a decision).
    """
    deg = _degree(grid)
    mask = deg >= 3
    if start is not None:
        sr, sc = start
        if grid[sr, sc] == 0 and deg[sr, sc] >= 2:
            mask[sr, sc] = True
    return mask


def dead_end_mask(grid: np.ndarray,
                  start: Tuple[int, int] | None = None,
                  goal: Tuple[int, int] | None = None) -> np.ndarray:
    """Path-cell mask of dead-ends — cells with degree == 1 (single
    passable neighbour). Start and goal are excluded even if their
    degree is 1, since they're terminals by design rather than traps."""
    deg = _degree(grid)
    mask = (deg == 1) & (grid == 0)
    if start is not None:
        mask[start] = False
    if goal is not None:
        mask[goal] = False
    return mask


def _bfs_dist(grid: np.ndarray, src: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid.shape
    dist = -np.ones_like(grid, dtype=np.int32)
    dist[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == 0 and dist[nr, nc] < 0):
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def junction_density(grid: np.ndarray,
                     start: Tuple[int, int] | None = None,
                     goal: Tuple[int, int] | None = None,
                     sigma: float = DENSITY_SIGMA) -> np.ndarray:
    """Graph-distance KDE summing kernels from BOTH junction cells AND
    dead-end cells. Each source contributes exp(-d_graph/sigma) at every
    reachable cell; walls block propagation."""
    rows, cols = grid.shape
    density = np.zeros((rows, cols), dtype=np.float64)
    sources = np.argwhere(
        junction_mask(grid, start=start)
        | dead_end_mask(grid, start=start, goal=goal)
    )
    if sources.size == 0:
        return density
    for sr, sc in sources:
        d = _bfs_dist(grid, (int(sr), int(sc)))
        kernel = np.exp(-d.astype(np.float64) / sigma)
        kernel[d < 0] = 0.0
        density += kernel
    density[grid == 1] = 0.0
    return density


# ---------------------------------------------------------------------------
# Per-panel rendering (preview-style maze + density-tinted path cells)
# ---------------------------------------------------------------------------

def _cell_center(r: int, c: int, rows: int) -> Tuple[float, float]:
    return (c + 0.5, (rows - 1 - r) + 0.5)


def render_panel(ax, maze: Dict, title: str, vmax: float,
                 cell_lw: float, radius: float, fontsize: float,
                 show_letters: bool, path_lw: float) -> None:
    grid = np.array(maze["grid"], dtype=np.int8)
    rows, cols = grid.shape
    start = tuple(maze["start"])
    goal  = tuple(maze["goal"])
    density = junction_density(grid, start=start, goal=goal)
    n_junctions = int(junction_mask(grid, start=start).sum())
    n_deadends  = int(dead_end_mask(grid, start=start, goal=goal).sum())

    # Per-cell facecolor: walls always light-grey; path cells take a tint
    # from the `Reds` cmap proportional to density.
    norm = density / vmax if vmax > 0 else density
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                fc = WALL_COLOR
            else:
                t = float(norm[r, c])
                fc = DENSITY_CMAP(t)
            ax.add_patch(Rectangle(
                (c, rows - 1 - r), 1, 1,
                facecolor=fc, edgecolor=GRID_COLOR, linewidth=cell_lw,
            ))

    # Optional shortest-path dashed overlay (kept for visual continuity with
    # the preview script).
    sp = maze.get("topology", {}).get("shortest_path", [])
    if sp:
        xs = [_cell_center(r, c, rows)[0] for r, c in sp]
        ys = [_cell_center(r, c, rows)[1] for r, c in sp]
        ax.plot(xs, ys, color=OPTIMAL_COLOR, linewidth=path_lw,
                linestyle=(0, (2, 2)), zorder=2)

    # S / G markers
    sr, sc = tuple(maze["start"]); gr, gc = tuple(maze["goal"])
    sx, sy = _cell_center(sr, sc, rows)
    gx, gy = _cell_center(gr, gc, rows)
    ax.add_patch(Circle((sx, sy), radius,
                        facecolor=START_COLOR, edgecolor="white",
                        linewidth=0.6, zorder=4))
    ax.add_patch(Circle((gx, gy), radius,
                        facecolor=GOAL_COLOR, edgecolor="white",
                        linewidth=0.6, zorder=4))
    if show_letters:
        ax.text(sx, sy, "S", ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold", zorder=5)
        ax.text(gx, gy, "G", ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold", zorder=5)

    ax.set_xlim(-0.1, cols + 0.1)
    ax.set_ylim(-0.1, rows + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(f"{title}  ·  junctions: {n_junctions}  ·  dead-ends: {n_deadends}",
                 fontsize=10, fontweight="bold", pad=4)


# ---------------------------------------------------------------------------
# Compose figure — one standalone SVG per difficulty tier (+ matching PNG)
# ---------------------------------------------------------------------------

def _render_single_svg(maze: dict, diff: str, vmax: float,
                       cell_lw: float, radius: float, fontsize: float,
                       show_letters: bool, path_lw: float,
                       filename_stem: str | None = None) -> None:
    """One maze → one stand-alone SVG (+ matching PNG), with its own
    inline colorbar. `vmax` is the GLOBAL max so all 3 SVGs share a
    colour scale and can be compared side-by-side outside the script.

    Filename: `heatmap_s5_{diff}.svg` by default, override via filename_stem.
    """
    fig = plt.figure(figsize=(4.2, 3.4))
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 0.05],
        wspace=0.12, left=0.05, right=0.84, top=0.88, bottom=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    render_panel(
        ax, maze, title=diff.capitalize(), vmax=vmax,
        cell_lw=cell_lw, radius=radius, fontsize=fontsize,
        show_letters=show_letters, path_lw=path_lw,
    )

    cax = fig.add_subplot(gs[0, 1])
    sm = mpl.cm.ScalarMappable(
        norm=Normalize(vmin=0, vmax=vmax), cmap=DENSITY_CMAP,
    )
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(
        f"Decision-point density  (junctions + dead-ends, σ={DENSITY_SIGMA})",
        fontsize=7, rotation=90, labelpad=6,
    )
    cbar.ax.tick_params(labelsize=6.5)

    stem = OUT_DIR / (filename_stem or f"heatmap_s5_{diff}")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved → {stem.relative_to(REPO)}.{{svg,png}}")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = load_mazes()

    grids   = [np.array(m["grid"], dtype=np.int8) for m in picks.values()]
    starts  = [tuple(m["start"]) for m in picks.values()]
    goals   = [tuple(m["goal"])  for m in picks.values()]
    global_max = float(max(
        junction_density(g, start=s, goal=go).max()
        for g, s, go in zip(grids, starts, goals)
    ))

    # Per-grid styling (mirrors render_maze_preview.py thresholds).
    rows = grids[0].shape[0]   # all 3 are 11×11 for s5
    if rows <= 11:
        cell_lw, radius, fontsize, path_lw = 0.5, 0.32, 8.0, 1.8
        show_letters = True
    elif rows <= 15:
        cell_lw, radius, fontsize, path_lw = 0.4, 0.34, 6.5, 1.4
        show_letters = True
    else:
        cell_lw, radius, fontsize, path_lw = 0.4, 0.45, 0.0, 1.2
        show_letters = False

    print(f"global_max density (shared across 3 SVGs) = {global_max:.4f}")
    for diff, m in picks.items():
        g = np.array(m["grid"], dtype=np.int8)
        s, go = tuple(m["start"]), tuple(m["goal"])
        njs = int(junction_mask(g, start=s).sum())
        nde = int(dead_end_mask(g, start=s, goal=go).sum())
        print(f"  {diff:6s} maze {m['id']}  junctions={njs}  dead-ends={nde}")
        _render_single_svg(m, diff, vmax=global_max,
                           cell_lw=cell_lw, radius=radius,
                           fontsize=fontsize, show_letters=show_letters,
                           path_lw=path_lw)


if __name__ == "__main__":
    main()
