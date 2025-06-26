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

import numpy as np
import pandas as pd
import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO, TASK_ORDER
from mrna_bench.utils import get_data_path
from scipy.stats import ttest_ind

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
# Z-Score and Significance Helpers
# -----------------------------------------------------------------------------

def _get_model_set(config_path: str, model_name: str, model_type: str) -> Set[str]:
    """Loads a set of models from a config file or a single name."""
    model_set = set()
    if model_name:
        model_set = {model_name}
        print(f"Using single model '{model_name}' as {model_type}.")
    elif config_path:
        model_set = _load_model_ckpt_config(config_path)
        print(f"Using {len(model_set)} models from {config_path} as {model_type}.")
    return model_set


def _calculate_reference_stats(
    df: pd.DataFrame,
    reference_models: Set[str]
) -> pd.DataFrame:
    """Calculate mean and std for a reference set of models."""
    ref_df = df[df['model'].isin(reference_models)]
    if ref_df.empty:
        raise ValueError("No data found for any of the reference models specified.")

    meta_cols = ["dataset", "model", "task", "target", "split", "seed"]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    group_cols = ["dataset", "task", "target", "split"]

    # Calculate mean and std over all seeds and all models in the reference set
    ref_stats = ref_df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()

    # Flatten the multi-level column index
    ref_stats.columns = ['_'.join(col).strip('_') for col in ref_stats.columns.values]
    return ref_stats


def _calculate_z_scores(
    df: pd.DataFrame,
    ref_stats: pd.DataFrame
) -> pd.DataFrame:
    """Calculate Z-scores for all models relative to reference stats."""
    group_cols = ["dataset", "task", "target", "split"]
    # Merge the reference stats back into the main dataframe
    df_with_stats = pd.merge(df, ref_stats, on=group_cols, how='left')

    meta_cols = ["dataset", "model", "task", "target", "split", "seed"]
    metric_cols = sorted([c for c in df.columns if c not in meta_cols])

    z_score_df = df_with_stats[meta_cols].copy()

    for metric in metric_cols:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'

        # Z-score = (value - mean) / std
        # Use np.divide to handle division by zero (std=0) gracefully -> results in inf
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.divide(
                df_with_stats[metric] - df_with_stats[mean_col],
                df_with_stats[std_col]
            )
        # Replace NaNs (from 0/0) and infs (from x/0) with 0, as it implies no deviation
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        z_score_df[f'{metric}_zscore'] = z_scores

    return z_score_df


def _run_significance_tests(
    z_score_df: pd.DataFrame,
    reference_models: Set[str],
    test_models: Set[str],
    zscore_cols: List[str]
) -> pd.DataFrame:
    """Performs t-test for each test model against a reference group of models."""

    if not test_models:
        print("[WARN] No test models specified via --config_file. Skipping significance tests.")
        return pd.DataFrame()

    ref_z = z_score_df[z_score_df['model'].isin(reference_models)]
    if ref_z.empty:
        print("[WARN] Not enough data for reference models for significance tests after filtering.")
        return pd.DataFrame()

    all_results = []
    # zscore_cols are now passed in directly
    group_cols = ["dataset", "task", "target", "split"]
    
    # Pre-group the reference dataframe for efficient lookup
    ref_z_grouped = ref_z.groupby(group_cols)

    # Iterate over each individual model in the "test" set
    for test_model_name in sorted(list(test_models)):
        test_z = z_score_df[z_score_df['model'] == test_model_name]
        
        if test_z.empty:
            continue

        # Group the current test model's data
        test_z_grouped = test_z.groupby(group_cols)

        # Iterate through the groups (tasks) for the current test model
        for group_key, test_group_df in test_z_grouped:
            try:
                # Find the matching group from the reference models
                ref_group_df = ref_z_grouped.get_group(group_key)
            except KeyError:
                continue # No reference data for this specific group

            for col in zscore_cols:
                # Ensure column exists in both dataframes for the group
                if col not in ref_group_df.columns or col not in test_group_df.columns:
                    continue

                ref_scores = ref_group_df[col].dropna()
                test_scores = test_group_df[col].dropna()

                if len(ref_scores) < 2 or len(test_scores) < 2:
                    continue # Not enough data for a t-test

                stat, pvalue = ttest_ind(test_scores, ref_scores, equal_var=False) # Welch's t-test
                
                result_row = dict(zip(group_cols, group_key))
                result_row['model'] = test_model_name # Identify which model is being tested
                result_row['metric'] = col.replace('_zscore', '')
                result_row['t_statistic'] = stat
                result_row['p_value'] = pvalue
                all_results.append(result_row)
    
    return pd.DataFrame(all_results)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _save_df_to_results(df: pd.DataFrame, output_filename: str):
    """Saves a dataframe to a file in the aggregated_results directory."""
    agg_results_dir = Path(get_data_path()) / "aggregated_results"
    agg_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = agg_results_dir / output_filename
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path.resolve()}")


def _get_parser() -> argparse.ArgumentParser:
    """Configures and returns the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Aggregate Orthrus linear-probe results across datasets.")
    parser.add_argument("--output_filename", type=str, default="",
                        help="Filename for the aggregated CSV summary, which will be saved in the mrna_bench data directory. If omitted, prints to STDOUT.")
    parser.add_argument("--config_file", type=str, default="",
                        help="Optional JSON/YAML file mapping model_version -> list[checkpoint] to limit aggregation (serves as the 'test set' for significance analysis).")
    parser.add_argument("--seeds", type=str, default="2541,413,411,412,2547,321,421,311,2516,2515",
                        help="Comma-separated list of seed integers to include in the aggregation. Example: 2541,413,421")
    parser.add_argument("--wide_format", action="store_true",
                        help="When set, pivot the output to have one row per model (wide format).")
    parser.add_argument("--no_aggregate_eclip", action="store_true", help="Show metrics for each eCLIP RBP individually instead of aggregating.")
    # --- Arguments for Z-score and significance testing ---
    parser.add_argument("--zscore_ref_config_file", type=str, default="",
                        help="JSON/YAML file for a *set of reference models* for Z-SCORE calculation. Mutually exclusive with --zscore_ref_model_name.")
    parser.add_argument("--zscore_ref_model_name", type=str, default="",
                        help="A single model_short_name to use as the reference for Z-SCORE calculation. Mutually exclusive with --zscore_ref_config_file.")
    parser.add_argument("--sig_ref_config_file", type=str, default="",
                        help="JSON/YAML file for the *reference model set* for SIGNIFICANCE testing. Mutually exclusive with --sig_ref_model_name.")
    parser.add_argument("--sig_ref_model_name", type=str, default="",
                        help="A single model_short_name to use as the reference for SIGNIFICANCE testing. Mutually exclusive with --sig_ref_config_file.")
    parser.add_argument("--z_score_output", type=str, default="",
                        help="Output filename for the Z-score report. Activates Z-score calculation.")
    parser.add_argument("--significance_output", type=str, default="",
                        help="Output filename for significance test results. Activates t-test calculation.")
    return parser


def _run_zscore_and_significance_analysis(df: pd.DataFrame, args: argparse.Namespace):
    """Orchestrates the Z-score and significance testing analysis."""
    print("--- Running Z-Score and Significance Analysis ---")
    try:
        # 1. Determine reference and test sets
        zscore_ref_models = _get_model_set(args.zscore_ref_config_file, args.zscore_ref_model_name, "reference for Z-Scores")
        sig_ref_models = _get_model_set(args.sig_ref_config_file, args.sig_ref_model_name, "reference for Significance Test")
        test_models = _get_model_set(args.config_file, "", "test models")

        if not zscore_ref_models:
            print("[ERROR] No reference models provided for Z-score calculation (--zscore_ref_*). Halting analysis.")
            return

        # 2. Loop through tasks to perform task-aware analysis
        tasks = df['task'].unique()
        all_pivoted_z_scores = []
        all_sig_results = []

        for task in tasks:
            task_df = df[df['task'] == task].copy()

            # Dynamically find metric columns for this task
            meta_cols = ["dataset", "model", "task", "target", "split", "seed"]
            task_df = task_df.dropna(axis=1, how='all')
            metric_cols = [c for c in task_df.columns if c not in meta_cols]
            if not metric_cols:
                continue
            
            # Calculate Z-scores for this task's metrics
            ref_stats = _calculate_reference_stats(task_df, zscore_ref_models)
            z_score_df_task = _calculate_z_scores(task_df, ref_stats)

            task_zscore_cols = [f"{m}_zscore" for m in metric_cols]

            # Generate Z-score report for the task
            if args.z_score_output:
                avg_z_scores = z_score_df_task.groupby(meta_cols[:-1])[task_zscore_cols].mean().reset_index()
                pivoted_z_score_task = _pivot_summary_wide(avg_z_scores)
                if not pivoted_z_score_task.empty:
                    all_pivoted_z_scores.append(pivoted_z_score_task)

            # Run significance tests for the task
            if args.significance_output and sig_ref_models:
                sig_results_task = _run_significance_tests(z_score_df_task, sig_ref_models, test_models, task_zscore_cols)
                if not sig_results_task.empty:
                    all_sig_results.append(sig_results_task)

        # 3. Combine and save final reports
        if args.z_score_output and all_pivoted_z_scores:
            final_z_score_df = reduce(lambda left, right: pd.merge(left, right, on='model', how='outer'), all_pivoted_z_scores)
            # Sort by model name and columns for consistency
            final_z_score_df = final_z_score_df.sort_values("model", ignore_index=True)
            if 'model' in final_z_score_df.columns:
                cols = final_z_score_df.columns.tolist()
                cols.remove('model')
                final_z_score_df = final_z_score_df[['model'] + sorted(cols)]
            
            _save_df_to_results(final_z_score_df, args.z_score_output)

        if args.significance_output and all_sig_results:
            final_sig_df = pd.concat(all_sig_results, ignore_index=True)
            # Sort for consistent output
            sort_cols = ['model', 'dataset', 'target', 'metric']
            final_sig_df = final_sig_df.sort_values(by=[c for c in sort_cols if c in final_sig_df.columns], ignore_index=True)

            _save_df_to_results(final_sig_df, args.significance_output)

    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] Could not run analysis: {e}")


def _run_standard_aggregation(df: pd.DataFrame, args: argparse.Namespace):
    """Orchestrates the standard metric aggregation and summarization."""
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
        _save_df_to_results(final_df, args.output_filename)
    else:
        # Print to STDOUT, stripping any trailing newline for cleaner piping
        print(final_df.to_csv(index=False).strip())


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if args.zscore_ref_config_file and args.zscore_ref_model_name:
        parser.error("argument --zscore_ref_config_file: not allowed with argument --zscore_ref_model_name")
    if args.sig_ref_config_file and args.sig_ref_model_name:
        parser.error("argument --sig_ref_config_file: not allowed with argument --sig_ref_model_name")

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

    # --- Apply Fisher Z-transformation to correlation coefficients ---
    # This stabilizes the variance of r/rho values for better averaging and Z-scoring.
    # Apply Fisher Z-transformation to correlation coefficient columns
    correlation_cols = ['test_r', 'train_r', 'val_r', 'test_p', 'train_p', 'val_p']
    for col in correlation_cols:
        if col in df.columns:
            # Clip to avoid infinity at -1 and 1, a common practice for this transformation
            df[col] = df[col].clip(-1 + 1e-9, 1 - 1e-9)
            df[col] = np.arctanh(df[col])
            print(f"Applied Fisher Z-transformation to '{col}' column.")
    # --- Aggregate eCLIP sub-tasks by default ---
    if not args.no_aggregate_eclip:
        # This now applies to both eCLIP and lncRNA essentiality datasets
        should_aggregate = df['dataset'].str.startswith('eclip-binding') | \
                           df['dataset'].str.startswith('lncrna-ess')

        if should_aggregate.any():
            df.loc[should_aggregate, 'target'] = 'aggregated'

    # ------------------------------------------------------------------
    # Z-Score and Significance Analysis (if activated)
    # ------------------------------------------------------------------
    if (args.zscore_ref_config_file or args.zscore_ref_model_name):
        _run_zscore_and_significance_analysis(df, args)

    # ------------------------------------------------------------------
    # Standard Aggregation
    # ------------------------------------------------------------------
    # This runs if any standard output is requested (wide format or file output)
    if args.wide_format or args.output_filename:
        _run_standard_aggregation(df, args)
    elif not (args.zscore_ref_config_file or args.zscore_ref_model_name):
        print("No analysis requested. Use --wide_format, --output_filename, or Z-score/significance flags.")


if __name__ == "__main__":
    main()

