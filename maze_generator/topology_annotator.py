"""Annotate maze topology: cell types, passability, shortest path, junction choices."""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .generator import PATH, WALL


@dataclass
class CellInfo:
    """Topology info for a single path cell."""
    row: int
    col: int
    passable: Dict[str, bool]  # {"up": bool, "down": bool, "left": bool, "right": bool}
    degree: int                # number of passable neighbors
    cell_type: str             # "dead-end", "corridor", "corner", "junction"


@dataclass
class JunctionChoice:
    """Info about a junction on the shortest path."""
    position: Tuple[int, int]              # (row, col)
    available_directions: List[str]        # all passable directions
    optimal_direction: str                 # direction on shortest path
    reachable_directions: List[str]        # directions that can reach goal
    dead_end_directions: List[str]         # directions leading to dead-ends
    dead_end_depths: Dict[str, int]        # direction -> depth of dead-end branch


@dataclass
class MazeAnnotation:
    """Complete topology annotation for a maze."""
    grid: np.ndarray
    effective_size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    cells: Dict[Tuple[int, int], CellInfo]
    shortest_path: List[Tuple[int, int]]
    shortest_path_length: int
    junctions_on_path: List[JunctionChoice]
    all_dead_ends: List[Tuple[int, int]]
    dead_end_branch_total_length: int


DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_passable(grid: np.ndarray, r: int, c: int) -> Dict[str, bool]:
    """Check which directions are passable from (r, c)."""
    rows, cols = grid.shape
    result = {}
    for name, (dr, dc) in DIRECTIONS.items():
        nr, nc = r + dr, c + dc
        result[name] = (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == PATH)
    return result


def _classify_cell(passable: Dict[str, bool]) -> Tuple[str, int]:
    """Classify cell type based on passable directions."""
    degree = sum(passable.values())
    if degree == 0:
        return "isolated", degree
    elif degree == 1:
        return "dead-end", degree
    elif degree == 2:
        dirs = [d for d, v in passable.items() if v]
        # Check if straight (opposite directions) or corner
        opposites = {("up", "down"), ("down", "up"), ("left", "right"), ("right", "left")}
        if (dirs[0], dirs[1]) in opposites:
            return "corridor", degree
        else:
            return "corner", degree
    else:
        return "junction", degree


def _bfs_shortest_path(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """BFS to find shortest path. Returns list of (row, col) or None."""
    rows, cols = grid.shape
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == PATH and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    return None


def _bfs_reachable(grid: np.ndarray, start: Tuple[int, int],
                   blocked: set) -> set:
    """BFS from start, not passing through blocked cells. Returns reachable cells."""
    rows, cols = grid.shape
    queue = deque([start])
    visited = {start}

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == PATH
                    and (nr, nc) not in visited
                    and (nr, nc) not in blocked):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return visited


def _measure_dead_end_branch(grid: np.ndarray, junction_pos: Tuple[int, int],
                             first_step: Tuple[int, int]) -> int:
    """Measure the depth of a branch starting from junction through first_step.

    Returns the number of cells in the branch if it's a dead-end branch
    (doesn't connect back to the maze except through junction_pos), else 0.
    """
    reachable = _bfs_reachable(grid, first_step, blocked={junction_pos})
    # If the branch reconnects to cells outside, it's not a pure dead-end branch
    # Check if reachable set includes any junction neighbor other than first_step's branch
    rows, cols = grid.shape
    for cell in reachable:
        r, c = cell
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if ((nr, nc) == junction_pos):
                continue
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == PATH
                    and (nr, nc) not in reachable):
                # This branch connects to cells outside the branch
                return 0

    return len(reachable)


def annotate_maze(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> MazeAnnotation:
    """Annotate a maze with full topology information.

    Args:
        grid: 2D numpy array (0=path, 1=wall)
        start: (row, col) of start cell
        goal: (row, col) of goal cell

    Returns:
        MazeAnnotation with cell types, shortest path, junction choices, etc.
    """
    rows, cols = grid.shape
    effective_size = (rows - 1) // 2

    # Annotate all path cells
    cells: Dict[Tuple[int, int], CellInfo] = {}
    all_dead_ends = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == PATH:
                passable = _get_passable(grid, r, c)
                cell_type, degree = _classify_cell(passable)
                cells[(r, c)] = CellInfo(
                    row=r, col=c, passable=passable,
                    degree=degree, cell_type=cell_type,
                )
                if cell_type == "dead-end":
                    all_dead_ends.append((r, c))

    # Shortest path via BFS
    shortest_path = _bfs_shortest_path(grid, start, goal)
    if shortest_path is None:
        raise ValueError(f"No path from {start} to {goal}")

    shortest_path_length = len(shortest_path) - 1  # number of steps

    # Analyze junctions on shortest path
    junctions_on_path = []
    path_set = set(shortest_path)

    for i, pos in enumerate(shortest_path):
        cell = cells[pos]
        if cell.cell_type != "junction":
            continue

        # Determine which direction the shortest path goes next
        if i + 1 < len(shortest_path):
            next_pos = shortest_path[i + 1]
            dr = next_pos[0] - pos[0]
            dc = next_pos[1] - pos[1]
            optimal_dir = {(-1, 0): "up", (1, 0): "down",
                           (0, -1): "left", (0, 1): "right"}[(dr, dc)]
        else:
            optimal_dir = ""

        # For each passable direction, check if it can reach the goal
        available = [d for d, v in cell.passable.items() if v]
        reachable_dirs = []
        dead_end_dirs = []
        dead_end_depths = {}

        for d in available:
            dr, dc = DIRECTIONS[d]
            next_cell = (pos[0] + dr, pos[1] + dc)
            # Can the goal be reached from next_cell without going through pos?
            reachable = _bfs_reachable(grid, next_cell, blocked={pos})
            if goal in reachable:
                reachable_dirs.append(d)
            else:
                dead_end_dirs.append(d)
                dead_end_depths[d] = len(reachable)

        junctions_on_path.append(JunctionChoice(
            position=pos,
            available_directions=available,
            optimal_direction=optimal_dir,
            reachable_directions=reachable_dirs,
            dead_end_directions=dead_end_dirs,
            dead_end_depths=dead_end_depths,
        ))

    # Total dead-end branch length
    dead_end_branch_total = sum(
        depth
        for jc in junctions_on_path
        for depth in jc.dead_end_depths.values()
    )

    return MazeAnnotation(
        grid=grid,
        effective_size=effective_size,
        start=start,
        goal=goal,
        cells=cells,
        shortest_path=shortest_path,
        shortest_path_length=shortest_path_length,
        junctions_on_path=junctions_on_path,
        all_dead_ends=all_dead_ends,
        dead_end_branch_total_length=dead_end_branch_total,
    )
