"""Four input formats for maze representation: Words, Coordinate, Map, Picture.

Each formatter takes a maze dict (as stored in data/mazes/*.json) and returns
a string prompt (or, for Picture, a list of message content blocks with base64 image).
"""

import base64
import io
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from maze_generator.generator import PATH, WALL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_array(maze: Dict[str, Any]) -> np.ndarray:
    """Convert maze['grid'] (list of lists) to numpy array."""
    g = maze["grid"]
    if isinstance(g, np.ndarray):
        return g
    return np.array(g, dtype=np.int8)


def _start_goal(maze: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    s, g = maze["start"], maze["goal"]
    return tuple(s), tuple(g)


# ---------------------------------------------------------------------------
# 1. Words format — row-by-row natural language (0-indexed)
# ---------------------------------------------------------------------------

def to_words(maze: Dict[str, Any]) -> str:
    grid = _grid_array(maze)
    start, goal = _start_goal(maze)
    rows, cols = grid.shape

    lines = [f"The maze is a {rows}x{cols} grid (row 0 = top, column 0 = left):"]
    for r in range(rows):
        cells = []
        for c in range(cols):
            cells.append("wall" if grid[r, c] == WALL else "path")
        lines.append(f"Row {r}: " + ", ".join(cells))

    sr, sc = start
    gr, gc = goal
    lines.append(f"Start: Row {sr}, Column {sc}. Goal: Row {gr}, Column {gc}.")
    lines.append("Find a path from Start to Goal.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Coordinate format — list of open cells (0-indexed)
# ---------------------------------------------------------------------------

def to_coordinate(maze: Dict[str, Any]) -> str:
    grid = _grid_array(maze)
    start, goal = _start_goal(maze)
    rows, cols = grid.shape

    open_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == PATH:
                open_cells.append(f"({r},{c})")

    lines = [
        f"The maze is a {rows}x{cols} grid (row 0 = top, column 0 = left).",
        "Open cells: " + ",".join(open_cells),
        f"Walls: all other cells in the grid.",
        f"Start: ({start[0]},{start[1]}). Goal: ({goal[0]},{goal[1]}).",
        "Find a path from Start to Goal. Output as a coordinate sequence.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Map format — ASCII text map
# ---------------------------------------------------------------------------

def to_map(maze: Dict[str, Any]) -> str:
    grid = _grid_array(maze)
    start, goal = _start_goal(maze)
    rows, cols = grid.shape

    lines = [f"The maze is a {rows}x{cols} grid (row 0 = top, column 0 = left):"]
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            pos = (r, c)
            if pos == start:
                row_chars.append("S")
            elif pos == goal:
                row_chars.append("G")
            elif grid[r, c] == WALL:
                row_chars.append("#")
            else:
                row_chars.append(".")
        lines.append(f"{r:2d}: " + "".join(row_chars))

    lines.append("'#' = wall, '.' = path, 'S' = start, 'G' = goal.")
    lines.append("Find a path from S to G.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Picture format — base64-encoded PNG image
# ---------------------------------------------------------------------------

def to_picture_base64(maze: Dict[str, Any], cell_size: int = 20) -> str:
    """Return base64-encoded PNG string of the maze image."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("Pillow is required for Picture format: pip install Pillow")

    grid = _grid_array(maze)
    start, goal = _start_goal(maze)
    rows, cols = grid.shape
    width, height = cols * cell_size, rows * cell_size

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            pos = (r, c)
            if pos == start:
                color = (66, 133, 244)    # blue
            elif pos == goal:
                color = (234, 67, 53)     # red
            elif grid[r, c] == WALL:
                color = (30, 30, 30)      # black
            else:
                color = (255, 255, 255)   # white
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_picture_message(maze: Dict[str, Any], cell_size: int = 20) -> List[Dict[str, Any]]:
    """Return a list of content blocks suitable for multimodal LLM API calls.

    Returns a list like:
    [
        {"type": "text", "text": "...instruction..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]
    """
    grid = _grid_array(maze)
    start, goal = _start_goal(maze)
    rows, cols = grid.shape
    b64 = to_picture_base64(maze, cell_size)
    return [
        {
            "type": "text",
            "text": (
                "The maze is shown in the image below. "
                "Black = wall, White = path, Blue = start, Red = goal.\n"
                f"The maze is a {rows}x{cols} grid. "
                "Row 0 is the TOP row, column 0 is the LEFT column. "
                f"Start: Blue({start[0]},{start[1]}). Goal: Red({goal[0]},{goal[1]}).\n"
                "Find a path from Start to Goal."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

FORMAT_NAMES = ("words", "coordinate", "map", "picture")

def format_maze(maze: Dict[str, Any], fmt: str, cell_size: int = 20) -> Union[str, List[Dict]]:
    """Convert a maze dict to the specified input format.

    Args:
        maze: Maze dictionary (as stored in JSON data files).
        fmt: One of 'words', 'coordinate', 'map', 'picture'.
        cell_size: Pixel size per cell (only used for 'picture').

    Returns:
        str for words/coordinate/map; list of content blocks for picture.
    """
    fmt = fmt.lower()
    if fmt == "words":
        return to_words(maze)
    elif fmt == "coordinate":
        return to_coordinate(maze)
    elif fmt == "map":
        return to_map(maze)
    elif fmt == "picture":
        return to_picture_message(maze, cell_size)
    else:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {FORMAT_NAMES}")
