"""Maze generation using DFS (Recursive Backtracker) and Prim's algorithm.

Maze representation:
  - numpy 2D array of shape (2*n+1, 2*n+1) where n = effective_size
  - 0 = path, 1 = wall
  - Passage cells are at odd coordinates: (2i+1, 2j+1) for i,j in [0, n-1]
  - Start: bottom-left (2n-1, 1), Goal: top-right (1, 2n-1)
"""

import numpy as np
from typing import Tuple

WALL = 1
PATH = 0


def _init_grid(n: int) -> np.ndarray:
    """Create a (2n+1) x (2n+1) grid filled with walls."""
    size = 2 * n + 1
    return np.ones((size, size), dtype=np.int8)


def _cell_to_grid(r: int, c: int) -> Tuple[int, int]:
    """Convert cell coordinates (0-based) to grid coordinates."""
    return 2 * r + 1, 2 * c + 1


def _get_neighbors(r: int, c: int, n: int) -> list:
    """Get valid cell neighbors (in cell coordinates)."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            neighbors.append((nr, nc))
    return neighbors


def generate_maze_dfs(effective_size: int, seed: int = 0) -> np.ndarray:
    """Generate maze using DFS / Recursive Backtracker.

    Produces long winding corridors with fewer branches.

    Args:
        effective_size: number of passage cells per side (e.g., 3 means 3x3 passages)
        seed: random seed for reproducibility

    Returns:
        numpy array of shape (2*effective_size+1, 2*effective_size+1)
    """
    n = effective_size
    rng = np.random.RandomState(seed)
    grid = _init_grid(n)
    visited = np.zeros((n, n), dtype=bool)

    # Start from a random cell
    start_r, start_c = rng.randint(0, n), rng.randint(0, n)
    visited[start_r, start_c] = True
    gr, gc = _cell_to_grid(start_r, start_c)
    grid[gr, gc] = PATH

    stack = [(start_r, start_c)]

    while stack:
        r, c = stack[-1]
        neighbors = [(nr, nc) for nr, nc in _get_neighbors(r, c, n)
                      if not visited[nr, nc]]

        if neighbors:
            nr, nc = neighbors[rng.randint(0, len(neighbors))]
            visited[nr, nc] = True

            # Carve passage cell
            gr, gc = _cell_to_grid(nr, nc)
            grid[gr, gc] = PATH

            # Carve wall between current and neighbor
            wr = _cell_to_grid(r, c)[0] + (nr - r)
            wc = _cell_to_grid(r, c)[1] + (nc - c)
            grid[wr, wc] = PATH

            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def generate_maze_prim(effective_size: int, seed: int = 0) -> np.ndarray:
    """Generate maze using randomized Prim's algorithm.

    Produces more branching structures with more dead-ends.

    Args:
        effective_size: number of passage cells per side
        seed: random seed for reproducibility

    Returns:
        numpy array of shape (2*effective_size+1, 2*effective_size+1)
    """
    n = effective_size
    rng = np.random.RandomState(seed)
    grid = _init_grid(n)
    visited = np.zeros((n, n), dtype=bool)

    # Start from a random cell
    start_r, start_c = rng.randint(0, n), rng.randint(0, n)
    visited[start_r, start_c] = True
    gr, gc = _cell_to_grid(start_r, start_c)
    grid[gr, gc] = PATH

    # Wall list: each entry is (cell_r, cell_c, neighbor_r, neighbor_c)
    # representing the wall between a visited cell and an unvisited neighbor
    wall_list = []
    for nr, nc in _get_neighbors(start_r, start_c, n):
        wall_list.append((start_r, start_c, nr, nc))

    while wall_list:
        idx = rng.randint(0, len(wall_list))
        cr, cc, nr, nc = wall_list[idx]
        wall_list[idx] = wall_list[-1]
        wall_list.pop()

        if visited[nr, nc]:
            continue

        visited[nr, nc] = True

        # Carve the neighbor cell
        gr, gc = _cell_to_grid(nr, nc)
        grid[gr, gc] = PATH

        # Carve the wall between
        wr = _cell_to_grid(cr, cc)[0] + (nr - cr)
        wc = _cell_to_grid(cr, cc)[1] + (nc - cc)
        grid[wr, wc] = PATH

        # Add new walls
        for nnr, nnc in _get_neighbors(nr, nc, n):
            if not visited[nnr, nnc]:
                wall_list.append((nr, nc, nnr, nnc))

    return grid


def get_start_goal(effective_size: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return start (bottom-left) and goal (top-right) in grid coordinates.

    Returns:
        (start, goal) as ((row, col), (row, col))
    """
    n = effective_size
    start = (2 * n - 1, 1)          # bottom-left passage
    goal = (1, 2 * n - 1)           # top-right passage
    return start, goal


# Mapping from effective size to grid size
EFFECTIVE_SIZES = [3, 5, 7, 10, 15, 20, 30]
GRID_SIZES = {n: 2 * n + 1 for n in EFFECTIVE_SIZES}  # {3:7, 5:11, ...}
