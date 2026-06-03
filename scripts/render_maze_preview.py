"""Render maze previews matching the visual style of the error-type
schematic (analysis/fig_error_types_schematic.py), with start = blue and
goal = red:

  - Light-grey walls  (#cfcfcf)
  - White path cells with thin grey grid borders  (#dddddd)
  - Grey dashed shortest-path overlay  (#aaaaaa)
  - Blue S marker (filled circle + "S" letter)
  - Red  G marker (filled circle + "G" letter)

Outputs PNGs to data/maze_previews/maze_s{N}.png for N in [3,5,7,10,15,20,30].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MAZE_DIR = REPO / "data" / "mazes"
OUT_DIR = REPO / "data" / "maze_previews"

# Style — mirror fig_error_types_schematic except S/G colors per request.
WALL_COLOR    = "#cfcfcf"
PATH_COLOR    = "#ffffff"
GRID_COLOR    = "#dddddd"
OPTIMAL_COLOR = "#aaaaaa"
START_COLOR   = "#3F70C8"   # blue
GOAL_COLOR    = "#E84B3A"   # red


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    })


def load_maze(size: int, difficulty: str = "medium", index: int = 0) -> dict:
    path = MAZE_DIR / f"mazes_s{size}.json"
    with open(path) as f:
        all_mazes = json.load(f)
    by_diff = [m for m in all_mazes if m.get("difficulty") == difficulty]
    return by_diff[index]


def _cell_center(r: int, c: int, rows: int) -> Tuple[float, float]:
    """Convert (row, col) to plot xy (cell center). Row 0 at the top."""
    return (c + 0.5, (rows - 1 - r) + 0.5)


def _draw_maze_grid(ax, grid: np.ndarray, cell_lw: float = 0.4) -> None:
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = WALL_COLOR if grid[r, c] == 1 else PATH_COLOR
            ax.add_patch(Rectangle(
                (c, rows - 1 - r), 1, 1,
                facecolor=color, edgecolor=GRID_COLOR, linewidth=cell_lw,
            ))


def _draw_optimal_path(ax, path: List[Tuple[int, int]],
                        rows: int, lw: float) -> None:
    if not path:
        return
    xs = [_cell_center(r, c, rows)[0] for r, c in path]
    ys = [_cell_center(r, c, rows)[1] for r, c in path]
    ax.plot(xs, ys, color=OPTIMAL_COLOR, linewidth=lw,
            linestyle=(0, (2, 2)), zorder=2)


def _draw_start_goal(ax, start: Tuple[int, int], goal: Tuple[int, int],
                      rows: int, radius: float, fontsize: float,
                      show_letters: bool = True) -> None:
    sx, sy = _cell_center(*start, rows)
    gx, gy = _cell_center(*goal, rows)
    ax.add_patch(Circle((sx, sy), radius,
                        facecolor=START_COLOR, edgecolor="white",
                        linewidth=0.6, zorder=4))
    ax.add_patch(Circle((gx, gy), radius,
                        facecolor=GOAL_COLOR, edgecolor="white",
                        linewidth=0.6, zorder=4))
    if show_letters:
        ax.text(sx, sy, "S", ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold",
                zorder=5)
        ax.text(gx, gy, "G", ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold",
                zorder=5)


def render(size: int, out_path: Path, target_in: float = 3.0) -> None:
    maze = load_maze(size, difficulty="medium", index=0)
    grid = np.array(maze["grid"])
    rows, cols = grid.shape

    # Scale path linewidth, marker radius, and "S"/"G" label size with grid
    # density so previews stay readable from s3 (small) to s30 (very dense).
    # For grids ≥ 21 (effective size ≥ 10) the S/G letters no longer fit in
    # the per-cell-sized dot, so we drop them and rely on solid colour alone.
    if rows <= 11:
        path_lw, radius, fontsize, cell_lw = 1.8, 0.32, 8.0, 0.5
        show_letters = True
    elif rows <= 15:
        path_lw, radius, fontsize, cell_lw = 1.4, 0.34, 6.5, 0.4
        show_letters = True
    elif rows <= 21:
        path_lw, radius, fontsize, cell_lw = 1.2, 0.45, 0.0, 0.4
        show_letters = False
    elif rows <= 31:
        path_lw, radius, fontsize, cell_lw = 0.9, 0.55, 0.0, 0.3
        show_letters = False
    else:
        path_lw, radius, fontsize, cell_lw = 0.6, 0.65, 0.0, 0.2
        show_letters = False

    fig, ax = plt.subplots(figsize=(target_in, target_in))
    _draw_maze_grid(ax, grid, cell_lw=cell_lw)
    _draw_optimal_path(
        ax, maze.get("topology", {}).get("shortest_path", []),
        rows, lw=path_lw,
    )
    _draw_start_goal(ax, tuple(maze["start"]), tuple(maze["goal"]),
                      rows, radius=radius, fontsize=fontsize,
                      show_letters=show_letters)

    ax.set_xlim(-0.1, cols + 0.1)
    ax.set_ylim(-0.1, rows + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in [3, 5, 7, 10, 15, 20, 30]:
        out = OUT_DIR / f"maze_s{s}.png"
        render(s, out)
        print(f"  s{s} -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
