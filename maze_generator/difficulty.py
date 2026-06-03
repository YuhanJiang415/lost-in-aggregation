"""Difficulty metrics computation and dataset generation."""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

from .generator import (
    generate_maze_dfs, generate_maze_prim,
    get_start_goal, EFFECTIVE_SIZES,
)
from .topology_annotator import annotate_maze, MazeAnnotation


@dataclass
class DifficultyMetrics:
    """Quantitative difficulty metrics for a maze."""
    junction_count_on_path: int     # number of junctions on shortest path
    dead_end_density: float         # num_dead_ends / total_path_cells
    confusion_ratio: float          # total_dead_end_branch_length / shortest_path_length
    shortest_path_length: int       # number of steps in shortest path
    total_path_cells: int           # total number of path cells
    total_dead_ends: int            # total number of dead-end cells
    composite_score: float = 0.0    # weighted composite for sorting


def compute_difficulty_metrics(annotation: MazeAnnotation) -> DifficultyMetrics:
    """Compute difficulty metrics from a maze annotation."""
    total_path_cells = len(annotation.cells)
    total_dead_ends = len(annotation.all_dead_ends)
    dead_end_density = total_dead_ends / total_path_cells if total_path_cells > 0 else 0
    spl = annotation.shortest_path_length
    confusion_ratio = annotation.dead_end_branch_total_length / spl if spl > 0 else 0

    junction_count = len(annotation.junctions_on_path)

    # Composite score: weighted sum for ranking
    # Normalize each metric relative to maze size
    composite = (
        0.4 * junction_count / max(spl, 1) +    # junction density on path
        0.3 * dead_end_density +                  # dead-end density
        0.3 * confusion_ratio / max(spl, 1)       # confusion relative to maze size
    )

    return DifficultyMetrics(
        junction_count_on_path=junction_count,
        dead_end_density=round(dead_end_density, 4),
        confusion_ratio=round(confusion_ratio, 4),
        shortest_path_length=spl,
        total_path_cells=total_path_cells,
        total_dead_ends=total_dead_ends,
        composite_score=round(composite, 6),
    )


def _classify_into_terciles(
    mazes: list, n_per_difficulty: int
) -> dict:
    """Sort mazes by composite_score and split into easy/medium/hard terciles."""
    sorted_mazes = sorted(mazes, key=lambda m: m["metrics"]["composite_score"])
    total = len(sorted_mazes)

    if total < 3 * n_per_difficulty:
        raise ValueError(
            f"Need at least {3 * n_per_difficulty} mazes, got {total}. "
            f"Increase pool_multiplier."
        )

    return {
        "easy": sorted_mazes[:n_per_difficulty],
        "medium": sorted_mazes[total // 2 - n_per_difficulty // 2:
                               total // 2 + (n_per_difficulty + 1) // 2],
        "hard": sorted_mazes[-n_per_difficulty:],
    }


def _maze_to_dict(
    maze_id: str,
    grid: np.ndarray,
    algorithm: str,
    effective_size: int,
    seed: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    annotation: MazeAnnotation,
    metrics: DifficultyMetrics,
) -> dict:
    """Convert maze data to a JSON-serializable dict."""
    # Cell types: only store type per cell (compact)
    cell_types = {
        f"{r},{c}": info.cell_type
        for (r, c), info in annotation.cells.items()
    }
    # Passability per cell (convert numpy bools to Python bools)
    passability = {
        f"{r},{c}": {k: bool(v) for k, v in info.passable.items()}
        for (r, c), info in annotation.cells.items()
    }
    # Junction choices on path
    junction_choices = []
    for jc in annotation.junctions_on_path:
        junction_choices.append({
            "position": list(jc.position),
            "available_directions": jc.available_directions,
            "optimal_direction": jc.optimal_direction,
            "reachable_directions": jc.reachable_directions,
            "dead_end_directions": jc.dead_end_directions,
            "dead_end_depths": jc.dead_end_depths,
        })

    return {
        "id": maze_id,
        "effective_size": effective_size,
        "grid_size": grid.shape[0],
        "algorithm": algorithm,
        "seed": seed,
        "grid": grid.tolist(),
        "start": list(start),
        "goal": list(goal),
        "metrics": asdict(metrics),
        "topology": {
            "cell_types": cell_types,
            "passability": passability,
            "shortest_path": [list(p) for p in annotation.shortest_path],
            "junction_choices": junction_choices,
            "dead_ends": [list(de) for de in annotation.all_dead_ends],
        },
    }


def generate_dataset(
    sizes: List[int] = None,
    n_per_difficulty: int = 50,
    pool_multiplier: int = 5,
    base_seed: int = 42,
    output_dir: str = "data/mazes",
) -> dict:
    """Generate the full maze dataset.

    For each size, generates a pool of mazes (half DFS, half Prim),
    computes difficulty metrics, and selects n_per_difficulty mazes
    for each of easy/medium/hard.

    Args:
        sizes: list of effective sizes. Defaults to EFFECTIVE_SIZES.
        n_per_difficulty: number of mazes per (size, difficulty, algorithm)
        pool_multiplier: generate this many times more mazes than needed for selection
        base_seed: starting random seed
        output_dir: directory to save JSON files

    Returns:
        Summary dict with counts and file paths.
    """
    if sizes is None:
        sizes = EFFECTIVE_SIZES

    os.makedirs(output_dir, exist_ok=True)
    summary = {"sizes": {}, "total_mazes": 0}

    for size in sizes:
        print(f"Generating mazes for effective_size={size}...")
        start, goal = get_start_goal(size)
        pool_per_algo = n_per_difficulty * pool_multiplier

        # Generate pool
        pool = []
        maze_counter = 0

        for algo_name, gen_fn in [("dfs", generate_maze_dfs),
                                   ("prim", generate_maze_prim)]:
            for i in range(pool_per_algo):
                seed = base_seed + size * 10000 + i + (5000 if algo_name == "prim" else 0)
                grid = gen_fn(size, seed=seed)
                annotation = annotate_maze(grid, start, goal)
                metrics = compute_difficulty_metrics(annotation)

                maze_id = f"maze_s{size}_{algo_name}_{maze_counter:04d}"
                maze_counter += 1

                maze_dict = _maze_to_dict(
                    maze_id, grid, algo_name, size, seed,
                    start, goal, annotation, metrics,
                )
                pool.append(maze_dict)

        # Split by algorithm, classify each into terciles, then merge
        dfs_pool = [m for m in pool if m["algorithm"] == "dfs"]
        prim_pool = [m for m in pool if m["algorithm"] == "prim"]

        n_half = n_per_difficulty // 2  # 25 DFS + 25 Prim per difficulty

        dfs_classified = _classify_into_terciles(dfs_pool, n_half)
        prim_classified = _classify_into_terciles(prim_pool, n_half)

        size_data = {}
        for difficulty in ["easy", "medium", "hard"]:
            selected = dfs_classified[difficulty] + prim_classified[difficulty]
            # Re-assign IDs and difficulty label
            for j, m in enumerate(selected):
                m["difficulty"] = difficulty
                m["id"] = f"maze_s{size}_{difficulty}_{j:03d}"

            size_data[difficulty] = selected

        # Save to file
        all_mazes = []
        for diff in ["easy", "medium", "hard"]:
            all_mazes.extend(size_data[diff])

        filepath = os.path.join(output_dir, f"mazes_s{size}.json")
        with open(filepath, "w") as f:
            json.dump(all_mazes, f, indent=2)

        count = len(all_mazes)
        summary["sizes"][size] = {"count": count, "file": filepath}
        summary["total_mazes"] += count
        print(f"  Size {size}: {count} mazes saved to {filepath}")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {summary['total_mazes']} mazes generated.")
    return summary
