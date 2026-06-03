"""Maze visualization: ASCII text and PNG image rendering."""

from typing import List, Optional, Tuple

import numpy as np

from .generator import PATH, WALL

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def maze_to_ascii(
    grid: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    current: Optional[Tuple[int, int]] = None,
) -> str:
    """Render maze as ASCII string.

    Args:
        grid: 2D numpy array (0=path, 1=wall)
        start: (row, col) to mark as 'S'
        goal: (row, col) to mark as 'G'
        path: list of (row, col) to mark as '*'
        current: (row, col) to mark as '@'

    Returns:
        Multi-line ASCII string
    """
    rows, cols = grid.shape
    path_set = set(path) if path else set()

    lines = []
    for r in range(rows):
        line = []
        for c in range(cols):
            pos = (r, c)
            if current and pos == current:
                line.append("@")
            elif start and pos == start:
                line.append("S")
            elif goal and pos == goal:
                line.append("G")
            elif pos in path_set:
                line.append("*")
            elif grid[r, c] == WALL:
                line.append("#")
            else:
                line.append(".")
        lines.append("".join(line))

    return "\n".join(lines)


def maze_to_image(
    grid: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    cell_size: int = 20,
) -> "Image.Image":
    """Render maze as a PIL Image.

    Args:
        grid: 2D numpy array (0=path, 1=wall)
        start: (row, col) to mark blue
        goal: (row, col) to mark red
        path: list of (row, col) to mark green
        cell_size: pixel size per cell

    Returns:
        PIL Image object
    """
    if not HAS_PIL:
        raise ImportError("Pillow is required: pip install Pillow")

    rows, cols = grid.shape
    width = cols * cell_size
    height = rows * cell_size
    path_set = set(path) if path else set()

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            pos = (r, c)
            if start and pos == start:
                color = (66, 133, 244)   # blue
            elif goal and pos == goal:
                color = (234, 67, 53)    # red
            elif pos in path_set:
                color = (52, 168, 83)    # green
            elif grid[r, c] == WALL:
                color = (30, 30, 30)     # dark gray / black
            else:
                color = (255, 255, 255)  # white

            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)

    return img


def save_maze_image(
    grid: np.ndarray,
    filepath: str,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    cell_size: int = 20,
) -> None:
    """Render and save maze as PNG."""
    img = maze_to_image(grid, start, goal, path, cell_size)
    img.save(filepath)
