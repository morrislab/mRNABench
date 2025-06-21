"""Aggregate Orthrus linear-probe results across all datasets.

This script scans every dataset registered in mrna_bench, locates JSON result
files produced by the linear-probe pipeline (``lp_results/`` sub-folder of each
dataset) and summarises metrics for every ``model_short_name``.

Output: a CSV file (or STDOUT) containing mean & std over random seeds for each
(metric, model, dataset, split, target, task) combination.

Usage
-----
python aggregate_orthrus_lp.py \
    --config_file ../model_json_files/test_dual_heads_and_underweight_20250616.json \
    --wide_format \
    --output_filename test_dual_heads.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Set
import yaml
from functools import reduce

import pandas as pd
import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO, TASK_ORDER
from mrna_bench.utils import get_data_path

# -----------------------------------------------------------------------------
# Helper for filename parsing
# -----------------------------------------------------------------------------

def _parse_filename(fname: str) -> Dict[str, Any] | None:
    """
    Parses a result filename to extract metadata components.

    Returns a dict with keys dataset, model, task, target, split, seed or
    ``None`` if the pattern does not match.
    """
    known_tasks = {"classification", "reg_ridge", "multilabel"}
    known_datasets = set(DATASET_INFO.keys())

    # --- 1. Initial Validation ---
    if not (fname.startswith("result_lp_") and fname.endswith(".json")):
        return None

    # --- 2. Primary Split ---
    # Use `_tcol-` as a stable anchor to divide the filename.
    core = fname[len("result_lp_") : -len(".json")]

    if "_tcol-" not in core:
        return None

    prefix, remainder = core.split("_tcol-", 1)

    # --- 3. Parse Right-Hand Side (target, split, seed) ---
    try:
        # Find the last `_rs-` to safely locate the seed.
        rs_idx = remainder.rfind("_rs-")
        if rs_idx == -1: return None
        seed = int(remainder[rs_idx + len("_rs-"):])

        # Everything before the seed part.
        before_seed = remainder[:rs_idx]

        # Find the last `_split-` to separate target and split.
        split_idx = before_seed.rfind("_split-")
        if split_idx == -1: return None
        target = before_seed[:split_idx]  # Allows underscores in target name.
        split = before_seed[split_idx + len("_split-"):]

    except (ValueError, IndexError):
        return None # Handles errors from rfind, slicing, or int conversion.

    # --- 4. Parse Left-Hand Side (dataset, model, task) ---
    # Find the task by checking for a known suffix.
    task = next((t for t in known_tasks if prefix.endswith(f"_{t}")), None)
    if task is None:
        return None
    prefix_without_task = prefix[: -len(f"_{task}")]

    # Find the dataset by checking for a known prefix.
    # Sorting by length ensures we match longer names first (e.g., 'pcg-ess-k562' before 'pcg-ess').
    dataset = next((d for d in sorted(known_datasets, key=len, reverse=True) if prefix_without_task.startswith(f"{d}_")), None)
    if dataset is None:
        return None

    # The remainder in the middle is the model name.
    model = prefix_without_task[len(f"{dataset}_"):]
    if not model:
        return None  # Model name should not be empty.

    # --- 5. Return Results ---
    return {
        "dataset": dataset,
        "model": model,
        "task": task,
        "target": target,
        "split": split,
        "seed": seed,
    }

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for ds_name, info in DATASET_INFO.items():
        dataset = mb.load_dataset(ds_name)
        result_dir = Path(dataset.dataset_path) / "lp_results"
        if not result_dir.exists():
            continue  # no results yet for this dataset

        for fp in result_dir.glob("result_lp_*.json"):
            meta = _parse_filename(fp.name)
            if meta is None:
                print(f"[WARN] Unrecognised filename pattern: {fp.name}")
                continue

            try:
                metrics = json.loads(fp.read_text())
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse JSON: {fp.name} ({e})")
                continue

            row: Dict[str, Any] = meta.copy()
            # Flatten numeric metrics into the row (skip nested dicts)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# Config loading utility (same format used elsewhere)
# -----------------------------------------------------------------------------

def _load_model_ckpt_config(cfg_path: str | Path) -> Set[str]:
    """Return a set of *model_short_name* strings derived from the mapping.

    The config file should map ``model_version -> list[checkpoint]`` (same as
    earlier scripts). We convert each pair into the canonical `model_short_name`
    format so that we can match it against filenames in ``lp_results``.
    """
    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML required to parse YAML config; install pyyaml.")
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())

    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping of model_version -> list[ckpt]")

    short_names: Set[str] = set()
    for mv, ckpts in data.items():
        if isinstance(ckpts, str):
            ckpts = [ckpts]
        for ck in ckpts:
            sn = (mv + "_" + ck.replace(".ckpt", "")).replace("_", "-").replace("-track", "").replace("best-", "")
            short_names.add(sn)
    return short_names


# -----------------------------------------------------------------------------
# Aggregation and Pivoting Helpers
# -----------------------------------------------------------------------------

def _process_task_group(task_df: pd.DataFrame) -> pd.DataFrame:
    """Processes a dataframe for a single task type.

    This function takes a DataFrame containing results for a single task (e.g.,
    'reg_ridge'), identifies the relevant metric columns by dropping any that
    are entirely null for that task, and then computes the mean and standard
    deviation over random seeds.

    Returns
    -------
    pd.DataFrame
        A long-format summary DataFrame for the task with mean/std metrics.
    """
    meta_cols = ["dataset", "model", "task", "target", "split", "seed"]

    # Dynamically find metric columns and drop any that are all NaN for this group
    all_cols = task_df.columns.tolist()
    task_df = task_df.dropna(axis=1, how='all')
    surviving_cols = task_df.columns.tolist()
    
    metric_cols = [c for c in surviving_cols if c not in meta_cols]

    if not metric_cols:
        return pd.DataFrame()

    # Compute mean & std over seeds
    group_cols = ["model", "dataset", "task", "target", "split"]
    grouped = task_df.groupby(group_cols)[metric_cols]
    mean_df = grouped.mean()
    std_df = grouped.std()

    # Suffix columns for clarity
    mean_df = mean_df.rename(columns={col: f"{col}_mean" for col in metric_cols})
    std_df = std_df.rename(columns={col: f"{col}_std" for col in metric_cols})

    # Combine mean and std, which are now aligned by the same index
    summary = mean_df.join(std_df)
    return summary.reset_index()


def _pivot_summary_wide(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots a long-format summary dataframe to be wide."""
    if summary_df.empty:
        return pd.DataFrame()

    pivot_cols = ['dataset', 'task', 'target', 'split']
    id_cols = ['model'] + pivot_cols
    metric_cols = [c for c in summary_df.columns if c not in id_cols]

    # Ensure columns have a consistent, sorted order
    metric_cols = sorted(metric_cols)

    # Use pivot_table to handle the restructuring
    pivoted = pd.pivot_table(
        summary_df,
        index='model',
        columns=pivot_cols,
        values=metric_cols,
        aggfunc='first'  # No aggregation needed, just restructuring
    )

    # Flatten the column MultiIndex and sort for deterministic output
    pivoted.columns = pivoted.columns.reorder_levels(
        [1, 2, 3, 4, 0] # dataset, task, target, split, metric
    )
    pivoted = pivoted.sort_index(axis=1)

    pivoted.columns = [
        f"{ds}_{tk}_{tgt}_{splt}_{met}"
        for ds, tk, tgt, splt, met in pivoted.columns
    ]
    
    return pivoted.reset_index()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate Orthrus linear-probe results across datasets.")
    parser.add_argument("--output_filename", type=str, default="",
                        help="Filename for the aggregated CSV summary, which will be saved in the mrna_bench data directory. If omitted, prints to STDOUT.")
    parser.add_argument("--config_file", type=str, default="",
                        help="Optional JSON/YAML file mapping model_version -> list[checkpoint] to limit aggregation.")
    parser.add_argument("--seeds", type=str, default="2541,413,411,412,2547,321,421,311,2516,2515",
                        help="Comma-separated list of seed integers to include in the aggregation. Example: 2541,413,421")
    parser.add_argument("--wide_format", action="store_true",
                        help="When set, pivot the output to have one row per model (wide format).")
    parser.add_argument("--no_aggregate_eclip", action="store_true", help="Show metrics for each eCLIP RBP individually instead of aggregating.")

    args = parser.parse_args()

    rows = _collect_rows()

    # ------------------------------------------------------------------
    # Optional filtering by models / seeds
    # ------------------------------------------------------------------

    if args.config_file:
        keep_models = _load_model_ckpt_config(args.config_file)
        rows = [r for r in rows if r["model"] in keep_models]

    if args.seeds:
        try:
            seed_set = {int(s) for s in args.seeds.split(",") if s.strip()}
        except ValueError:
            raise ValueError("--seeds must be a comma-separated list of integers")
        rows = [r for r in rows if r["seed"] in seed_set]

    if not rows:
        print("No rows left after applying filters – nothing to aggregate.")
        return

    df = pd.DataFrame(rows)
    # --- Aggregate eCLIP sub-tasks by default ---
    if not args.no_aggregate_eclip:
        # This now applies to both eCLIP and lncRNA essentiality datasets
        should_aggregate = df['dataset'].str.startswith('eclip-binding') | \
                           df['dataset'].str.startswith('lncrna-ess')

        if should_aggregate.any():
            df.loc[should_aggregate, 'target'] = 'aggregated'

    # ------------------------------------------------------------------
    # Process each task type independently to handle different metrics
    # ------------------------------------------------------------------
    tasks = df['task'].unique()
    all_summaries = []
    for task in tasks:
        task_df = df[df['task'] == task].copy()
        summary = _process_task_group(task_df)
        if not summary.empty:
            all_summaries.append(summary)

    if not all_summaries:
        print("No data left after processing task groups.")
        return

    if args.wide_format:
        # ------------------------------------------------------------------
        # Pivot each task's summary and merge into a single wide dataframe
        # ------------------------------------------------------------------
        all_pivoted_dfs = [_pivot_summary_wide(s) for s in all_summaries]

        # Merge all pivoted dataframes on the 'model' column
        final_df = reduce(
            lambda left, right: pd.merge(left, right, on='model', how='outer'),
            all_pivoted_dfs
        )
        # Sort by model name and columns for consistency
        final_df = final_df.sort_values("model", ignore_index=True)
        
        # Make 'model' the first column, then sort the rest alphabetically
        if 'model' in final_df.columns:
            cols = final_df.columns.tolist()
            cols.remove('model')
            final_df = final_df[['model'] + sorted(cols)]

    else:
        # ------------------------------------------------------------------
        # Combine long-format summaries and apply canonical sorting
        # ------------------------------------------------------------------
        final_df = pd.concat(all_summaries, ignore_index=True)

        # Stable row ordering: dataset and task order follows the catalog
        final_df['canonical_target'] = final_df['dataset'] + ":" + final_df['target']
        final_df['canonical_target'] = pd.Categorical(
            final_df['canonical_target'],
            categories=TASK_ORDER,
            ordered=True
        )
        # Sort and drop the temporary column
        final_df = final_df.sort_values(
            ['canonical_target', 'model', 'split'],
            ignore_index=True
        ).drop(columns='canonical_target')

    # --- Output ---
    if args.output_filename:
        # Get the base data path from mrna_bench and create a dedicated folder
        agg_results_dir = Path(get_data_path()) / "aggregated_results"
        agg_results_dir.mkdir(parents=True, exist_ok=True)
        out_path = agg_results_dir / args.output_filename
        final_df.to_csv(out_path, index=False)
        print(f"Aggregated results written to {out_path.resolve()}")
    else:
        # Print to STDOUT, stripping any trailing newline for cleaner piping
        print(final_df.to_csv(index=False).strip())


if __name__ == "__main__":
    main()

