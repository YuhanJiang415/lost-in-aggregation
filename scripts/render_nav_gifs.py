"""Animated maze-navigation GIFs for the project homepage.

Base map: the decision-point-density heatmap from `render_maze_heatmap.py`
(grey walls, white→deep-red path cells by junction/dead-end density, blue S /
red G, faint dashed ground-truth solution path). A little robot sprite
(`robot.png`, rasterized from `robot.svg`) walks the maze.

Scenarios (all on the SAME s5 maze so the correct route is a fixed reference):
  - success        : robot follows the unique solution to the goal.
  - err_wall_hit   : Fine error — robot lunges into a wall and bounces.
  - err_teleport   : Fine error — robot jumps to a non-adjacent cell.
  - err_wrong_turn : Meso error — robot takes the wrong branch at a junction
                     and dies in a dead-end.

Output: docs/assets/gifs/nav_{scenario}.gif
Run:    python3 scripts/render_nav_gifs.py
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

# Reuse the exact density model + styling from the heatmap script.
import importlib.util
import sys

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "render_maze_heatmap", REPO / "scripts" / "render_maze_heatmap.py")
hm = importlib.util.module_from_spec(_spec)
sys.modules["render_maze_heatmap"] = hm
_spec.loader.exec_module(hm)

MAZE_FILE = REPO / "mazes" / "mazes_s5.json"
ROBOT_PNG = REPO / "docs" / "assets" / "robot.png"
OUT_DIR = REPO / "docs" / "assets" / "gifs"

WALL_COLOR = "#cfcfcf"
GRID_COLOR = "#dddddd"
OPTIMAL_COLOR = "#9aa7b8"
START_COLOR = "#3F70C8"
GOAL_COLOR = "#E84B3A"
TRAIL_COLOR = "#1296db"      # matches the robot's blue
BAD_COLOR = "#E84B3A"
GOOD_COLOR = "#2e9e5b"
DENSITY_CMAP = plt.cm.Reds

# ---- timing ----
SUB = 7          # interpolation frames per single cell step
HOLD_START = 6
HOLD_END = 16
FPS = 18


# ---------------------------------------------------------------------------
# Maze / graph helpers
# ---------------------------------------------------------------------------

def load_maze(maze_id: str) -> dict:
    with open(MAZE_FILE) as f:
        mazes = json.load(f)
    for m in mazes:
        if m["id"] == maze_id:
            return m
    raise KeyError(maze_id)


def open_neighbors(grid: np.ndarray, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c = cell
    R, C = grid.shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr, nc] == 0:
            out.append((nr, nc))
    return out


def cell_center(r: int, c: int, rows: int) -> Tuple[float, float]:
    return (c + 0.5, (rows - 1 - r) + 0.5)


def walk_branch(grid, first_step, came_from, limit=40):
    """Greedily follow a corridor from `first_step` (entered from `came_from`)
    until a dead-end, never reversing. Returns the list of cells walked
    (including first_step)."""
    path = [first_step]
    prev, cur = came_from, first_step
    while len(path) < limit:
        nxt = [n for n in open_neighbors(grid, cur) if n != prev]
        if not nxt:
            break
        step = nxt[0]
        path.append(step)
        prev, cur = cur, step
    return path


# ---------------------------------------------------------------------------
# Move plan: a sequence of typed segments → expanded into per-frame states
# ---------------------------------------------------------------------------

def build_plan(maze: dict, scenario: str):
    grid = np.array(maze["grid"], dtype=np.int8)
    start = tuple(maze["start"])
    goal = tuple(maze["goal"])
    sp = [tuple(c) for c in maze["topology"]["shortest_path"]]
    deg = hm._degree(grid)

    if scenario == "success":
        return {"walk": sp, "error": None, "good": True}

    if scenario == "err_wall_hit":
        k = max(2, len(sp) // 3)
        cell = sp[k]
        nxt_on_path = sp[k + 1] if k + 1 < len(sp) else None
        came = sp[k - 1]
        # an adjacent WALL that is not the path direction
        r, c = cell
        wall = None
        for dr, dc in ((0, 1), (-1, 0), (0, -1), (1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 1:
                wall = (nr, nc)
                break
        return {"walk": sp[:k + 1], "bump": wall, "error": "wall", "good": False}

    if scenario == "err_teleport":
        k = max(2, len(sp) // 3)
        src = sp[k]
        # a far open cell (BFS distance >= 4), prefer a decoy off the path
        dist = hm._bfs_dist(grid, src)
        spset = set(sp)
        cands = [(int(r), int(c)) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                 if grid[r, c] == 0 and dist[r, c] >= 4 and (r, c) not in spset]
        if not cands:
            cands = [(int(r), int(c)) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                     if grid[r, c] == 0 and dist[r, c] >= 4]
        cands.sort(key=lambda x: -dist[x[0], x[1]])
        target = cands[len(cands) // 2] if cands else sp[-1]
        return {"walk": sp[:k + 1], "teleport": target, "error": "teleport", "good": False}

    if scenario == "err_wrong_turn":
        # first junction on the path with a wrong branch available
        for i in range(1, len(sp) - 1):
            cell = sp[i]
            if deg[cell] >= 3:
                came, correct = sp[i - 1], sp[i + 1]
                wrong = [n for n in open_neighbors(grid, cell)
                         if n != came and n != correct]
                if wrong:
                    branch = walk_branch(grid, wrong[0], cell)
                    return {"walk": sp[:i + 1], "wrong_from": cell,
                            "correct_next": correct, "wrong_walk": branch,
                            "error": "deadend", "good": False}
        # fallback: no junction → degrade to wall hit
        return build_plan(maze, "err_wall_hit")

    raise ValueError(scenario)


# ---------------------------------------------------------------------------
# Frame-state expansion
# ---------------------------------------------------------------------------

def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def expand_frames(maze, plan):
    """Return a list of frame-states. Each state:
      pos=(x,y), alpha, trail=[(x,y)...], marker=None|(kind,(x,y)),
      arc=None|((x,y),(x,y)), result=None|('good'|'bad', text)
    """
    grid = np.array(maze["grid"], dtype=np.int8)
    rows = grid.shape[0]
    xy = lambda cell: cell_center(cell[0], cell[1], rows)

    frames = []
    trail = []           # list of (x,y) the robot has occupied (cell centers)
    walk = plan["walk"]

    def emit(pos, alpha=1.0, marker=None, arc=None, result=None):
        frames.append(dict(pos=pos, alpha=alpha, trail=list(trail),
                           marker=marker, arc=arc, result=result))

    # start hold
    p0 = xy(walk[0])
    trail.append(p0)
    for _ in range(HOLD_START):
        emit(p0)

    # main walk along correct path
    for i in range(1, len(walk)):
        a, b = xy(walk[i - 1]), xy(walk[i])
        for s in range(1, SUB + 1):
            emit(lerp(a, b, s / SUB))
        trail.append(b)

    last = xy(walk[-1])

    if plan.get("bump"):                       # ---- wall hit ----
        wall_xy = xy(plan["bump"])
        for t in (0.18, 0.34, 0.46, 0.46, 0.30, 0.14, 0.0):
            emit(lerp(last, wall_xy, t))
        burst = ("wall", lerp(last, wall_xy, 0.5))
        for _ in range(HOLD_END):
            emit(last, marker=burst, result=("bad", "Wall hit  ·  Fine"))

    elif plan.get("teleport"):                 # ---- teleport ----
        tgt = xy(plan["teleport"])
        # fade out (trail still ends at the pre-jump cell)
        for a in (0.75, 0.45, 0.18):
            emit(last, alpha=a, arc=(last, tgt))
        # break the trail across the jump, then land at the target
        trail.append(None)
        trail.append(tgt)
        for a in (0.18, 0.5, 0.85, 1.0):
            emit(tgt, alpha=a, arc=(last, tgt))
        for _ in range(HOLD_END):
            emit(tgt, marker=("teleport", tgt), arc=(last, tgt),
                 result=("bad", "Teleport  ·  Fine"))

    elif plan.get("wrong_walk"):               # ---- wrong junction ----
        jxy = last
        cxy = xy(plan["correct_next"])
        branch = plan["wrong_walk"]
        # show the correct branch as a green arrow while turning wrong
        prev = jxy
        for i, cell in enumerate(branch):
            b = xy(cell)
            for s in range(1, SUB + 1):
                emit(lerp(prev, b, s / SUB), arc=("good_arrow", (jxy, cxy)))
            trail.append(b)
            prev = b
        dead = prev
        for _ in range(HOLD_END):
            emit(dead, marker=("deadend", dead),
                 arc=("good_arrow", (jxy, cxy)),
                 result=("bad", "Wrong junction  ·  Meso"))

    else:                                      # ---- success ----
        for _ in range(HOLD_END):
            emit(last, result=("good", "Reached the goal"))

    return frames


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_base(ax, maze, scenario_label):
    grid = np.array(maze["grid"], dtype=np.int8)
    rows, cols = grid.shape
    start = tuple(maze["start"])
    goal = tuple(maze["goal"])

    density = hm.junction_density(grid, start=start, goal=goal)
    vmax = float(density.max()) or 1.0
    norm = density / vmax

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                fc = WALL_COLOR
            else:
                fc = DENSITY_CMAP(float(norm[r, c]))
            ax.add_patch(Rectangle((c, rows - 1 - r), 1, 1,
                                   facecolor=fc, edgecolor=GRID_COLOR,
                                   linewidth=0.5, zorder=1))

    sp = maze["topology"]["shortest_path"]
    xs = [cell_center(r, c, rows)[0] for r, c in sp]
    ys = [cell_center(r, c, rows)[1] for r, c in sp]
    ax.plot(xs, ys, color=OPTIMAL_COLOR, lw=1.6, ls=(0, (2, 2)), zorder=2)

    for cell, col, letter in ((start, START_COLOR, "S"), (goal, GOAL_COLOR, "G")):
        x, y = cell_center(cell[0], cell[1], rows)
        ax.add_patch(Circle((x, y), 0.30, facecolor=col, edgecolor="white",
                            linewidth=1.0, zorder=3))
        ax.text(x, y, letter, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)

    ax.set_xlim(-0.1, cols + 0.1)
    ax.set_ylim(-0.1, rows + 0.7)   # headroom for the label chip
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)

    ax.text(0.02, 0.985, scenario_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold", color="#1a1d24",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e4e8ef", lw=1),
            zorder=10)


def render_gif(maze, scenario, label, out_path):
    robot = plt.imread(ROBOT_PNG)
    plan = build_plan(maze, scenario)
    frames = expand_frames(maze, plan)

    fig, ax = plt.subplots(figsize=(3.7, 3.9), dpi=130)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    draw_base(ax, maze, label)

    robot_im = ax.imshow(robot, extent=(0, 1, 0, 1), zorder=8,
                         interpolation="bilinear")
    trail_line, = ax.plot([], [], color=TRAIL_COLOR, lw=3.0, alpha=0.55,
                          solid_capstyle="round", zorder=5)
    dyn = {"marker": None, "arc": None, "result": None}

    rsz = 0.40  # robot half-size in cells

    def clear_dyn():
        for key in ("marker", "arc", "result"):
            obj = dyn[key]
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                dyn[key] = None

    imgs = []
    for st in frames:
        x, y = st["pos"]
        robot_im.set_extent((x - rsz, x + rsz, y - rsz, y + rsz))
        robot_im.set_alpha(st["alpha"])

        if st["trail"]:
            tx, ty = [], []
            for p in st["trail"]:
                if p is None:          # pen-up: break the polyline (teleport)
                    tx.append(np.nan); ty.append(np.nan)
                else:
                    tx.append(p[0]); ty.append(p[1])
            tx.append(x); ty.append(y)
            trail_line.set_data(tx, ty)

        clear_dyn()

        arc = st["arc"]
        if arc is not None:
            kind = arc[0]
            if kind == "good_arrow":
                (jx, jy), (cx, cy) = arc[1]
                dyn["arc"] = ax.add_patch(FancyArrowPatch(
                    (jx, jy), (cx, cy), arrowstyle="-|>", mutation_scale=13,
                    color=GOOD_COLOR, lw=2.0, ls="--", zorder=6,
                    shrinkA=6, shrinkB=6))
            else:
                a, b = arc
                dyn["arc"] = ax.add_patch(FancyArrowPatch(
                    a, b, arrowstyle="-|>", mutation_scale=12,
                    connectionstyle="arc3,rad=0.25", color=BAD_COLOR,
                    lw=1.8, ls=(0, (3, 2)), zorder=6))

        mk = st["marker"]
        if mk is not None:
            kind, (mx, my) = mk
            if kind == "wall":
                dyn["marker"] = ax.scatter([mx], [my], s=340, marker="X",
                                           color=BAD_COLOR, zorder=9,
                                           linewidths=0)
            elif kind == "teleport":
                dyn["marker"] = ax.scatter([mx], [my], s=300, marker="*",
                                           color=BAD_COLOR, zorder=9,
                                           edgecolors="white", linewidths=0.8)
            elif kind == "deadend":
                dyn["marker"] = ax.scatter([mx], [my], s=340, marker="X",
                                           color=BAD_COLOR, zorder=9,
                                           linewidths=0)

        res = st["result"]
        if res is not None:
            kind, text = res
            col = GOOD_COLOR if kind == "good" else BAD_COLOR
            sym = "✓" if kind == "good" else "✗"
            dyn["result"] = ax.text(
                0.5, 0.03, f"{sym}  {text}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.45", fc=col, ec="none"),
                zorder=11)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(buf[..., :3].copy())

    plt.close(fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, imgs, fps=FPS, loop=0)
    print(f"  {scenario:16s} {len(imgs):3d} frames -> {out_path.relative_to(REPO)}")


def main():
    maze = load_maze("maze_s5_medium_000")
    scenarios = [
        ("success", "Solving the maze"),
        ("err_wrong_turn", "Failure mode: wrong turn"),
        ("err_wall_hit", "Failure mode: wall hit"),
        ("err_teleport", "Failure mode: teleport"),
    ]
    for scen, label in scenarios:
        render_gif(maze, scen, label, OUT_DIR / f"nav_{scen}.gif")


if __name__ == "__main__":
    main()
