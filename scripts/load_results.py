"""Common utilities for loading experiment results from JSONL files."""

import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[Dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_experiment_dir(exp_dir: Path) -> pd.DataFrame:
    """Load all results_s*.jsonl in a directory into a single DataFrame."""
    records = []
    for fpath in sorted(exp_dir.glob("results_s*.jsonl")):
        records.extend(load_jsonl(fpath))
    if not records:
        raise FileNotFoundError(f"No results_s*.jsonl found in {exp_dir}")
    return pd.json_normalize(records, sep=".")


def find_latest_run(exp_name: str) -> Path:
    """Find the latest timestamped run directory for an experiment."""
    base = PROJECT_ROOT / "results" / exp_name
    if not base.exists():
        raise FileNotFoundError(f"No results directory: {base}")

    # Check for timestamped subdirs
    subdirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name,
    )
    if subdirs:
        return subdirs[-1]

    # Legacy: results directly in exp_name dir
    if list(base.glob("results_s*.jsonl")):
        return base

    raise FileNotFoundError(f"No result runs found in {base}")


def load_experiment(
    exp_name: str,
    run_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load experiment results into DataFrame.

    Args:
        exp_name: e.g. 'exp1', 'exp2_oneshot'
        run_dir: Specific run directory path. If None, uses latest.

    Returns:
        DataFrame with flattened metrics columns like 'metrics.overall.SR'
    """
    if run_dir:
        exp_dir = Path(run_dir)
    else:
        exp_dir = find_latest_run(exp_name)
    print(f"Loading from: {exp_dir}")
    return load_experiment_dir(exp_dir)


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

OVERALL_METRICS = ["SR", "PE", "FES", "VMR", "SDTW", "NE"]
FINE_METRICS = ["WCR", "TR", "PA"]
MESO_METRICS = ["JA", "OJA", "DER", "BSR", "LR"]
MACRO_METRICS = ["MPR", "DDR"]

ALL_DIMENSION_METRICS = {
    "overall": OVERALL_METRICS,
    "fine": FINE_METRICS,
    "meso": MESO_METRICS,
    "macro": MACRO_METRICS,
}


def metric_col(dimension: str, metric: str) -> str:
    """Return the DataFrame column name for a metric, e.g. 'metrics.overall.SR'."""
    return f"metrics.{dimension}.{metric}"
