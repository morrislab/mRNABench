from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrna_bench.datasets import DATASET_CATALOG


WEBSITE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = WEBSITE_ROOT.parent
SOURCE_PATH = REPOSITORY_ROOT / "paper" / "leaderboard_results.parquet"
CONFIG_PATH = WEBSITE_ROOT / "leaderboard.config.json"
OUTPUT_DIR = WEBSITE_ROOT / "public" / "data"
JSON_PATH = OUTPUT_DIR / "leaderboard.json"
CSV_PATH = OUTPUT_DIR / "leaderboard-results.csv"

REQUIRED_COLUMNS = {
    "model",
    "model_group",
    "dataset_group",
    "dataset",
    "task",
    "target",
    "split",
    "canonical_split",
    "seed",
    "test_auprc",
    "test_r",
    "pre_z",
    "mean",
    "std",
    "z_score",
}

DATASET_LABELS = {
    "eclip-binding-hepg2": "eCLIP Binding (HepG2)",
    "eclip-binding-k562": "eCLIP Binding (K562)",
    "mirna-target": "miRNA Binding",
    "mrl-hl-lbkwk": "Paired MRL and RNA Half-Life",
    "mrl-sample-designed": "MRL MPRA (Designed)",
    "mrl-sample-egfp": "MRL MPRA (eGFP)",
    "mrl-sample-mcherry": "MRL MPRA (mCherry)",
    "mrl-sample-varying": "MRL MPRA (Varying Length)",
    "mrl-sugimoto": "Mean Ribosome Load (Sugimoto)",
    "rna-lifecycle-ietswaart": "RNA Lifecycle",
    "rna-loc-fazal": "RNA Subcellular Localization",
    "rnahl-human": "RNA Half-Life (Human)",
    "rnahl-mouse": "RNA Half-Life (Mouse)",
    "translation-efficiency-human": "Translation Efficiency (Human)",
    "translation-efficiency-mouse": "Translation Efficiency (Mouse)",
    "utr-variants-bohn-utr3": "Curated 3' UTR Variants",
    "utr-variants-bohn-utr5": "Curated 5' UTR Variants",
    "vep-traitgym-complex": "TraitGym VEP (Complex Traits)",
    "vep-traitgym-mendelian": "TraitGym VEP (Mendelian)",
}

TASK_LABELS = {
    "classification": "Classification",
    "multilabel": "Multilabel",
    "regression": "Regression",
}

TASK_ORDER = ("classification", "multilabel", "regression")

MODEL_ALIASES = {
    "orthrus-large-6-splice-only-10cl-randseed0": "orthrus+mlm",
}

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def finite(value: float | int | None, digits: int = 8) -> float | int | None:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return None
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def t_critical_95(n_runs: int) -> float:
    degrees_of_freedom = n_runs - 1
    if degrees_of_freedom < 1:
        return math.nan
    if degrees_of_freedom <= 30:
        return T_CRITICAL_95[degrees_of_freedom]
    return 1.96


def validate_source_z_scores(frame: pd.DataFrame) -> float:
    """Validate the submitted Fisher transform and z-score columns."""
    regression_rows = ~frame["task"].isin({
        "classification",
        "multilabel",
    })
    raw_value = frame["test_auprc"].where(
        ~regression_rows,
        frame["test_r"],
    )
    scored_rows = raw_value.notna()
    standardized = frame.loc[
        scored_rows,
        ["pre_z", "mean", "std", "z_score"],
    ]
    if (
        standardized.isna().any().any()
        or not np.isfinite(standardized.to_numpy()).all()
    ):
        raise ValueError(
            "Submitted standardization columns contain null or "
            "non-finite values."
        )

    invalid_correlations = (
        regression_rows
        & scored_rows
        & raw_value.abs().ge(1)
    )
    if invalid_correlations.any():
        raise ValueError(
            "Regression metrics must be strictly between -1 and 1 "
            "before the Fisher transform."
        )
    expected_pre_z = raw_value.loc[scored_rows].copy()
    regression_scored = regression_rows.loc[scored_rows]
    expected_pre_z.loc[regression_scored] = np.arctanh(
        raw_value.loc[scored_rows & regression_rows]
    )
    observed_pre_z = frame.loc[scored_rows, "pre_z"]
    if not np.allclose(
        expected_pre_z,
        observed_pre_z,
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            "Submitted pre_z values do not match raw metrics and the "
            "required Fisher transform."
        )

    standardization_group = ["seed", "dataset", "target", "split"]
    groupers = [
        frame.loc[scored_rows, column]
        for column in standardization_group
    ]
    expected_mean = expected_pre_z.groupby(groupers).transform("mean")
    expected_std = expected_pre_z.groupby(groupers).transform("std")
    if expected_std.isna().any() or expected_std.le(0).any():
        raise ValueError(
            "Every z-score group must contain multiple distinct model "
            "performances."
        )
    if not np.allclose(
        expected_mean,
        frame.loc[scored_rows, "mean"],
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            "Submitted standardization means do not match the source rows."
        )
    if not np.allclose(
        expected_std,
        frame.loc[scored_rows, "std"],
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            "Submitted standard deviations do not match the source rows."
        )

    expected_z_score = (expected_pre_z - expected_mean) / expected_std
    z_score_difference = (
        expected_z_score - frame.loc[scored_rows, "z_score"]
    ).abs()
    max_difference = float(z_score_difference.max())
    if max_difference > 1e-7:
        raise ValueError(
            "Submitted z_score does not match per-seed, dataset, target, "
            f"and split standardization: {max_difference:.8f}."
        )
    return max_difference


def normalize_rows(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(
            "Leaderboard parquet is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = source.copy()
    frame["model"] = frame["model"].replace(MODEL_ALIASES)
    frame.loc[frame["model"] == "orthrus+mlm", "model_group"] = "Orthrus"
    initial_rows = len(frame)
    default_splits = {
        dataset_id: dataset_class.METADATA.default_split_type
        for dataset_id, dataset_class in DATASET_CATALOG.items()
    }
    unknown_datasets = sorted(set(frame["dataset"]).difference(default_splits))
    if unknown_datasets:
        raise ValueError(
            "Leaderboard parquet contains datasets missing from DATASET_CATALOG: "
            + ", ".join(unknown_datasets)
        )

    frame["default_split_type"] = frame["dataset"].map(default_splits)
    default_mismatch_rows = int(
        (frame["canonical_split"] != frame["default_split_type"]).sum()
    )
    frame = frame[frame["split"] == frame["default_split_type"]].copy()
    default_split_rows = len(frame)
    max_z_score_difference = validate_source_z_scores(frame)

    excluded_datasets = set(config["excluded_datasets"])
    excluded_models = set(config["excluded_models"])
    frame = frame[~frame["dataset"].isin(excluded_datasets)]
    after_dataset_filter = len(frame)

    supported_tasks = {
        "classification",
        "multilabel",
        "reg_ridge",
        "regression",
        "regression_ridge",
        "regression_ols",
    }
    before_task_filter = len(frame)
    frame = frame[frame["task"].isin(supported_tasks)].copy()
    unsupported_task_rows = before_task_filter - len(frame)

    frame["legacy_task"] = frame["task"]
    frame["source_group"] = frame["dataset_group"]
    frame["biological_task"] = frame["task"].map(
        {
            "classification": "classification",
            "multilabel": "multilabel",
            "reg_ridge": "regression",
            "regression": "regression",
            "regression_ridge": "regression",
            "regression_ols": "regression",
        }
    )
    frame["result_task"] = frame["task"].map(
        {
            "classification": "classification",
            "multilabel": "multilabel",
            "reg_ridge": "regression_ridge",
            "regression": "regression_ridge",
            "regression_ridge": "regression_ridge",
            "regression_ols": "regression_ols",
        }
    )
    frame["evaluation_method"] = "linear_probe"
    frame["estimator"] = frame["task"].map(
        {
            "classification": "logistic_regression",
            "multilabel": "multioutput_logistic_regression",
            "reg_ridge": "ridge_cv_legacy",
            "regression": "ridge_cv_legacy",
            "regression_ridge": "ridge_cv",
            "regression_ols": "ordinary_least_squares",
        }
    )
    frame["metric_id"] = frame["task"].map(
        {
            "classification": "auprc",
            "multilabel": "auprc_micro",
            "reg_ridge": "pearson_r",
            "regression": "pearson_r",
            "regression_ridge": "pearson_r",
            "regression_ols": "pearson_r",
        }
    )
    regression_result_task = config["regression_result_task"]
    valid_regression_tasks = {"regression_ols", "regression_ridge"}
    if regression_result_task not in valid_regression_tasks:
        raise ValueError(
            "regression_result_task must be one of: "
            + ", ".join(sorted(valid_regression_tasks))
        )
    before_protocol_filter = len(frame)
    frame = frame[
        (frame["biological_task"] != "regression")
        | (frame["result_task"] == regression_result_task)
    ].copy()
    excluded_regression_protocol_rows = before_protocol_filter - len(frame)

    protocol_counts = (
        frame.groupby("biological_task")[
            ["evaluation_method", "result_task", "estimator"]
        ]
        .apply(lambda group: len(group.drop_duplicates()))
    )
    mixed_protocols = protocol_counts[protocol_counts > 1]
    if not mixed_protocols.empty:
        raise ValueError(
            "Leaderboard data mixes evaluation protocols within task groups: "
            + ", ".join(mixed_protocols.index)
        )

    frame["value"] = frame["test_auprc"].where(
        frame["biological_task"] != "regression",
        frame["test_r"],
    )
    before_null_filter = len(frame)
    frame = frame[
        frame["value"].notna()
        & frame["z_score"].notna()
    ].copy()
    null_metric_rows = before_null_filter - len(frame)

    before_model_filter = len(frame)
    frame = frame[~frame["model"].isin(excluded_models)].copy()
    after_model_filter = len(frame)

    configured_priority = config["legacy_regression_source_priority"]
    priority = {
        task: index
        for index, task in enumerate(configured_priority)
    } | {
        "regression_ridge": 0,
        "regression_ols": 0,
        "classification": 0,
        "multilabel": 0,
    }
    collision_key = [
        "model",
        "dataset",
        "target",
        "split",
        "seed",
        "result_task",
    ]
    collision_spreads = (
        frame.groupby(collision_key)["value"]
        .agg(lambda values: float(values.max() - values.min()))
    )
    max_collision_difference = float(collision_spreads.max())
    allowed_difference = float(
        config["legacy_regression_max_abs_difference"]
    )
    if max_collision_difference > allowed_difference:
        raise ValueError(
            "Legacy result collision exceeds configured tolerance: "
            f"{max_collision_difference:.8f} > {allowed_difference:.8f}"
        )
    material_collision_count = int((collision_spreads > 1e-6).sum())

    frame["task_priority"] = frame["legacy_task"].map(priority)
    if frame["task_priority"].isna().any():
        missing_priority = sorted(
            set(frame.loc[frame["task_priority"].isna(), "legacy_task"])
        )
        raise ValueError(
            "Missing source-task priority for: "
            + ", ".join(missing_priority)
        )
    before_collision_filter = len(frame)
    frame = (
        frame.sort_values(collision_key + ["task_priority"])
        .drop_duplicates(collision_key, keep="first")
        .drop(columns=["task_priority"])
    )
    duplicate_rows = before_collision_filter - len(frame)

    expected_dataset_ids = set(DATASET_LABELS).difference(
        excluded_datasets
    )
    observed_dataset_ids = set(frame["dataset"])
    missing_datasets = sorted(
        expected_dataset_ids.difference(observed_dataset_ids)
    )
    unexpected_datasets = sorted(
        observed_dataset_ids.difference(expected_dataset_ids)
    )
    if missing_datasets or unexpected_datasets:
        raise ValueError(
            "Leaderboard dataset universe mismatch. Missing: {}; "
            "unexpected: {}.".format(
                ", ".join(missing_datasets) or "none",
                ", ".join(unexpected_datasets) or "none",
            )
        )

    expected_pairs = {
        (dataset_id, target)
        for dataset_id in expected_dataset_ids
        for target in DATASET_CATALOG[dataset_id].METADATA.target_col
    }
    observed_pairs = set(
        frame[["dataset", "target"]].itertuples(
            index=False,
            name=None,
        )
    )
    missing_pairs = sorted(expected_pairs.difference(observed_pairs))
    unexpected_pairs = sorted(observed_pairs.difference(expected_pairs))
    if missing_pairs or unexpected_pairs:
        missing_text = ", ".join(
            f"{dataset}:{target}"
            for dataset, target in missing_pairs
        )
        unexpected_text = ", ".join(
            f"{dataset}:{target}"
            for dataset, target in unexpected_pairs
        )
        raise ValueError(
            "Leaderboard target universe mismatch. Missing: {}; "
            "unexpected: {}.".format(
                missing_text or "none",
                unexpected_text or "none",
            )
        )

    stats = {
        "source_rows": initial_rows,
        "default_split_rows": default_split_rows,
        "legacy_default_mismatch_rows": default_mismatch_rows,
        "excluded_dataset_rows": default_split_rows - after_dataset_filter,
        "excluded_model_rows": before_model_filter - after_model_filter,
        "excluded_unsupported_task_rows": unsupported_task_rows,
        "excluded_regression_protocol_rows": excluded_regression_protocol_rows,
        "excluded_null_metric_rows": null_metric_rows,
        "resolved_legacy_collision_rows": duplicate_rows,
        "material_legacy_collision_keys": material_collision_count,
        "max_legacy_collision_difference": max_collision_difference,
        "max_z_score_difference": max_z_score_difference,
        "retained_seed_rows": len(frame),
    }
    return frame, stats


def aggregate_units(
    frame: pd.DataFrame,
    expected_seeds: list[int],
    data_status: str,
) -> pd.DataFrame:
    expected_seed_set = set(expected_seeds)
    expected_runs = len(expected_seeds)
    unit_columns = [
        "model",
        "model_group",
        "source_group",
        "dataset",
        "target",
        "biological_task",
        "result_task",
        "evaluation_method",
        "estimator",
        "split",
        "default_split_type",
        "metric_id",
    ]
    units = (
        frame.groupby(unit_columns, dropna=False)
        .agg(
            mean=("value", "mean"),
            std_sample=("value", "std"),
            n_runs=("z_score", "count"),
            mean_z_score=("z_score", "mean"),
            std_z_score=("z_score", "std"),
        )
        .reset_index()
    )
    seeds = (
        frame.groupby(unit_columns, dropna=False)["seed"]
        .agg(lambda values: sorted({int(value) for value in values}))
        .reset_index(name="seeds")
    )
    units = units.merge(seeds, on=unit_columns, how="left", validate="one_to_one")
    source_tasks = (
        frame.groupby(unit_columns, dropna=False)["legacy_task"]
        .agg(lambda values: sorted(set(values)))
        .reset_index(name="source_tasks")
    )
    units = units.merge(
        source_tasks,
        on=unit_columns,
        how="left",
        validate="one_to_one",
    )
    critical_values = units["n_runs"].map(t_critical_95)
    margin = (
        critical_values
        * units["std_sample"]
        / units["n_runs"].pow(0.5)
    )
    units["ci95_low"] = units["mean"] - margin
    units["ci95_high"] = units["mean"] + margin
    units["coverage_status"] = units["seeds"].map(
        lambda seeds: (
            "complete"
            if set(seeds) == expected_seed_set
            else "incomplete"
        )
    )
    units["expected_runs"] = expected_runs
    units["split_role"] = "test"
    units["higher_is_better"] = True
    units["data_status"] = data_status

    return units


def build_dataset_scores(
    frame: pd.DataFrame,
    units: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "model",
        "model_group",
        "source_group",
        "dataset",
        "biological_task",
        "metric_id",
    ]
    dataset_seed_scores = (
        frame.groupby([*keys, "seed"])
        .agg(score=("z_score", "mean"))
        .reset_index()
    )
    scores = (
        dataset_seed_scores.groupby(keys)
        .agg(score=("score", "mean"))
        .reset_index()
    )
    unit_summary = (
        units.groupby(keys)
        .agg(
            raw_mean=("mean", "mean"),
            completed_units=("coverage_status", lambda values: (values == "complete").sum()),
            unit_count=("target", "count"),
        )
        .reset_index()
    )
    scores = scores.merge(
        unit_summary,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    expected = (
        units.groupby(["dataset", "biological_task"])["target"]
        .nunique()
        .reset_index(name="expected_units")
    )
    scores = scores.merge(
        expected,
        on=["dataset", "biological_task"],
        how="left",
        validate="many_to_one",
    )
    scores["coverage_ratio"] = scores["unit_count"] / scores["expected_units"]
    scores["rank_eligible"] = (
        (scores["unit_count"] == scores["expected_units"])
        & (scores["completed_units"] == scores["expected_units"])
    )
    scores["rank"] = pd.NA
    for dataset, indexes in scores.groupby("dataset").groups.items():
        eligible = scores.loc[indexes]
        eligible = eligible[eligible["rank_eligible"]]
        scores.loc[eligible.index, "rank"] = eligible["score"].rank(
            method="min",
            ascending=False,
        )
    return scores


def build_source_scores(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average targets and sub-datasets within each seed."""
    dataset_seed_scores = (
        frame.groupby([
            "model",
            "model_group",
            "biological_task",
            "source_group",
            "dataset",
            "seed",
        ])
        .agg(score=("z_score", "mean"))
        .reset_index()
    )
    source_seed_scores = (
        dataset_seed_scores.groupby([
            "model",
            "model_group",
            "biological_task",
            "source_group",
            "seed",
        ])
        .agg(
            score=("score", "mean"),
            dataset_count=("dataset", "count"),
        )
        .reset_index()
    )
    source_scores = (
        source_seed_scores.groupby(
            ["model", "model_group", "biological_task", "source_group"]
        )
        .agg(
            score=("score", "mean"),
            dataset_count=("dataset_count", "max"),
        )
        .reset_index()
    )
    return source_scores, source_seed_scores


def build_task_scores(
    units: pd.DataFrame,
    source_scores: pd.DataFrame,
    source_seed_scores: pd.DataFrame,
) -> pd.DataFrame:
    expected = (
        units.groupby("biological_task")
        .agg(
            expected_units=("target", lambda values: len(set(values))),
            expected_datasets=("dataset", "nunique"),
            expected_source_groups=("source_group", "nunique"),
        )
        .reset_index()
    )
    # Targets can share names across datasets, so count exact dataset-target pairs.
    pair_counts = (
        units[["biological_task", "dataset", "target"]]
        .drop_duplicates()
        .groupby("biological_task")
        .size()
    )
    expected["expected_units"] = expected["biological_task"].map(pair_counts)

    observed_units = (
        units.groupby(["model", "biological_task"])
        .agg(
            unit_count=("target", "count"),
            completed_units=("coverage_status", lambda values: (values == "complete").sum()),
        )
        .reset_index()
    )
    task_seed_scores = (
        source_seed_scores.groupby([
            "model",
            "model_group",
            "biological_task",
            "seed",
        ])
        .agg(score=("score", "mean"))
        .reset_index()
    )
    task_score_values = (
        task_seed_scores.groupby(
            ["model", "model_group", "biological_task"]
        )
        .agg(score=("score", "mean"))
        .reset_index()
    )
    task_coverage = (
        source_scores.groupby(["model", "model_group", "biological_task"])
        .agg(
            source_group_count=("source_group", "count"),
            dataset_count=("dataset_count", "sum"),
        )
        .reset_index()
    )
    task_scores = (
        task_score_values.merge(
            task_coverage,
            on=["model", "model_group", "biological_task"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            observed_units,
            on=["model", "biological_task"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            expected,
            on="biological_task",
            how="left",
            validate="many_to_one",
        )
    )
    task_scores["coverage_ratio"] = (
        task_scores["unit_count"] / task_scores["expected_units"]
    )
    task_scores["rank_eligible"] = (
        (task_scores["unit_count"] == task_scores["expected_units"])
        & (task_scores["completed_units"] == task_scores["expected_units"])
        & (task_scores["dataset_count"] == task_scores["expected_datasets"])
        & (
            task_scores["source_group_count"]
            == task_scores["expected_source_groups"]
        )
    )
    task_scores["rank"] = pd.NA
    for task, indexes in task_scores.groupby("biological_task").groups.items():
        eligible = task_scores.loc[indexes]
        eligible = eligible[eligible["rank_eligible"]]
        task_scores.loc[eligible.index, "rank"] = eligible["score"].rank(
            method="min",
            ascending=False,
        )
    return task_scores


def build_overall_ranking(
    task_scores: pd.DataFrame,
    source_seed_scores: pd.DataFrame,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    """Rank complete models by mean z-score across source datasets."""
    eligible_tasks = task_scores[
        task_scores["rank_eligible"]
        & task_scores["biological_task"].isin(TASK_ORDER)
    ]
    rank_matrix = eligible_tasks.pivot(
        index="model",
        columns="biological_task",
        values="rank",
    ).dropna(subset=list(TASK_ORDER))
    task_score_matrix = eligible_tasks.pivot(
        index="model",
        columns="biological_task",
        values="score",
    )
    complete_sources = source_seed_scores[
        source_seed_scores["model"].isin(rank_matrix.index)
    ]
    overall_seed_scores = (
        complete_sources.groupby(["model", "model_group", "seed"])
        .agg(
            mean_z_score=("score", "mean"),
            source_group_count=("source_group", "count"),
        )
        .reset_index()
    )
    expected_source_groups = source_seed_scores["source_group"].nunique()
    overall_seed_scores = overall_seed_scores[
        overall_seed_scores["source_group_count"] == expected_source_groups
    ]
    complete_seed_models = (
        overall_seed_scores.groupby("model")["seed"]
        .agg(lambda seeds: set(seeds) == set(expected_seeds))
    )
    complete_seed_models = set(
        complete_seed_models[complete_seed_models].index
    )
    overall_seed_scores = overall_seed_scores[
        overall_seed_scores["model"].isin(complete_seed_models)
    ]
    overall_scores = (
        overall_seed_scores.groupby(["model", "model_group"])
        .agg(mean_z_score=("mean_z_score", "mean"))
        .reset_index()
    ).copy()
    overall_scores["rank"] = overall_scores["mean_z_score"].rank(
        method="min",
        ascending=False,
    )
    overall_scores = overall_scores.sort_values(
        ["rank", "mean_z_score", "model"],
        ascending=[True, False, True],
    )

    rows: list[dict[str, Any]] = []
    for _, overall in overall_scores.iterrows():
        model_id = overall["model"]
        task_ranks = {
            task: int(rank_matrix.loc[model_id, task])
            for task in TASK_ORDER
        }
        task_scores_for_model = {
            task: finite(task_score_matrix.loc[model_id, task])
            for task in TASK_ORDER
        }
        rows.append(
            {
                "model_id": model_id,
                "model_family": overall["model_group"],
                "rank": int(overall["rank"]),
                "mean_z_score": finite(overall["mean_z_score"]),
                "task_ranks": task_ranks,
                "task_scores": task_scores_for_model,
                "coverage_ratio": 1.0,
            }
        )
    return rows


def build_payload(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    normalized, row_stats = normalize_rows(source, config)
    expected_seeds = [
        int(seed)
        for seed in config["expected_seeds"]
    ]
    if (
        len(expected_seeds) != int(config["expected_runs"])
        or len(set(expected_seeds)) != len(expected_seeds)
    ):
        raise ValueError(
            "expected_seeds must contain expected_runs unique values."
        )
    units = aggregate_units(
        normalized,
        expected_seeds=expected_seeds,
        data_status=str(config["data_status"]),
    )
    dataset_scores = build_dataset_scores(normalized, units)
    source_scores, source_seed_scores = build_source_scores(normalized)
    task_scores = build_task_scores(
        units,
        source_scores,
        source_seed_scores,
    )
    overall_ranking = build_overall_ranking(
        task_scores,
        source_seed_scores,
        expected_seeds=expected_seeds,
    )

    task_rankings: dict[str, list[dict[str, Any]]] = {}
    for task in TASK_ORDER:
        rows = task_scores[task_scores["biological_task"] == task].copy()
        rows = rows.sort_values(
            ["rank_eligible", "rank", "score", "model"],
            ascending=[False, True, False, True],
            na_position="last",
        )
        task_rankings[task] = [
            {
                "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
                "model_id": row["model"],
                "model_family": row["model_group"],
                "score": finite(row["score"]),
                "coverage_ratio": finite(row["coverage_ratio"]),
                "dataset_count": int(row["dataset_count"]),
                "expected_datasets": int(row["expected_datasets"]),
                "source_group_count": int(row["source_group_count"]),
                "expected_source_groups": int(row["expected_source_groups"]),
                "rank_eligible": bool(row["rank_eligible"]),
            }
            for _, row in rows.iterrows()
        ]

    dataset_rankings: dict[str, list[dict[str, Any]]] = {}
    for dataset_id, rows in dataset_scores.groupby("dataset"):
        rows = rows.sort_values(
            ["rank_eligible", "rank", "score", "model"],
            ascending=[False, True, False, True],
            na_position="last",
        )
        dataset_rankings[dataset_id] = [
            {
                "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
                "model_id": row["model"],
                "model_family": row["model_group"],
                "score": finite(row["score"]),
                "raw_mean": finite(row["raw_mean"]),
                "metric_id": row["metric_id"],
                "coverage_ratio": finite(row["coverage_ratio"]),
                "target_count": int(row["unit_count"]),
                "expected_targets": int(row["expected_units"]),
                "rank_eligible": bool(row["rank_eligible"]),
            }
            for _, row in rows.iterrows()
        ]

    result_rows = []
    for _, row in units.sort_values(
        ["dataset", "target", "model"]
    ).iterrows():
        result_rows.append(
            {
                "model_id": row["model"],
                "model_family": row["model_group"],
                "dataset_id": row["dataset"],
                "source_group": row["source_group"],
                "result_release_id": config["release_id"],
                "target_col": row["target"],
                "biological_task": row["biological_task"],
                "result_task": row["result_task"],
                "evaluation_method": row["evaluation_method"],
                "estimator": row["estimator"],
                "split_type": row["split"],
                "split_role": "test",
                "metric_id": row["metric_id"],
                "higher_is_better": True,
                "n_runs": int(row["n_runs"]),
                "expected_runs": int(config["expected_runs"]),
                "mean": finite(row["mean"]),
                "std_sample": finite(row["std_sample"]),
                "mean_z_score": finite(row["mean_z_score"]),
                "std_z_score": finite(row["std_z_score"]),
                "ci95_low": finite(row["ci95_low"]),
                "ci95_high": finite(row["ci95_high"]),
                "coverage_status": row["coverage_status"],
                "data_status": row["data_status"],
                "seeds": row["seeds"],
                "source_tasks": row["source_tasks"],
            }
        )

    datasets = []
    for dataset_id in sorted(dataset_rankings):
        dataset_units = units[units["dataset"] == dataset_id]
        task = dataset_units["biological_task"].iloc[0]
        metric_id = dataset_units["metric_id"].iloc[0]
        datasets.append(
            {
                "id": dataset_id,
                "label": DATASET_LABELS.get(dataset_id, dataset_id),
                "biological_task": task,
                "task_label": TASK_LABELS[task],
                "metric_id": metric_id,
                "target_count": int(dataset_units["target"].nunique()),
                "source_group": dataset_units["source_group"].iloc[0],
                "split_type": dataset_units["split"].iloc[0],
            }
        )

    models = [
        {
            "id": model_id,
            "family": group["model_group"].iloc[0],
        }
        for model_id, group in units.groupby("model")
    ]

    payload = {
        "schema_version": "1.2.0",
        "source": {
            "artifact": SOURCE_PATH.name,
            "artifact_sha256": file_sha256(SOURCE_PATH),
            "release": config["release_id"],
            "label": config["release_label"],
            "status": config["data_status"],
            "package_commit": os.environ.get("GITHUB_SHA"),
            "effective_default_splits": {
                dataset["id"]: next(
                    row["split_type"]
                    for row in result_rows
                    if row["dataset_id"] == dataset["id"]
                )
                for dataset in datasets
            },
        },
        "methodology": {
            "split_policy": "Each dataset's declared default split is used for every model.",
            "standardization": "Metrics are z-scored across models for each seed, dataset, and target; Pearson correlations are Fisher-transformed first.",
            "seed_aggregation": "Z-scores are averaged across seeds after target and sub-dataset aggregation.",
            "expected_seeds": expected_seeds,
            "regression_result_task": config["regression_result_task"],
            "dataset_weighting": "Targets are averaged within each dataset so multi-target datasets do not dominate.",
            "source_weighting": "Registered dataset IDs are averaged within source dataset groups.",
            "task_weighting": "Source-dataset z-scores are averaged within each prediction task.",
            "overall": "Complete models are ranked by mean z-score across source datasets.",
        },
        "row_stats": row_stats,
        "exclusions": {
            "datasets": config["excluded_datasets"],
            "models": config["excluded_models"],
            "zero_shot": "This artifact contains no zero-shot result rows.",
        },
        "tasks": [
            {"id": task, "label": TASK_LABELS[task]}
            for task in TASK_ORDER
            if task in task_scores["biological_task"].unique()
        ],
        "datasets": datasets,
        "models": models,
        "rankings": {
            "overall": overall_ranking,
            **task_rankings,
        },
        "dataset_rankings": dataset_rankings,
        "results": result_rows,
    }
    return payload, units


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Leaderboard source was not found: {SOURCE_PATH}"
        )

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    source = pd.read_parquet(SOURCE_PATH)
    payload, units = build_payload(source, config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

    csv_columns = [
        "model",
        "model_group",
        "source_group",
        "dataset",
        "target",
        "biological_task",
        "result_task",
        "evaluation_method",
        "estimator",
        "split",
        "metric_id",
        "higher_is_better",
        "source_tasks",
        "n_runs",
        "expected_runs",
        "mean",
        "std_sample",
        "mean_z_score",
        "std_z_score",
        "ci95_low",
        "ci95_high",
        "coverage_status",
        "split_role",
        "data_status",
    ]
    csv_frame = units[csv_columns].rename(
        columns={
            "model": "model_id",
            "model_group": "model_family",
            "dataset": "dataset_id",
            "target": "target_col",
            "split": "split_type",
        }
    )
    csv_frame["source_tasks"] = csv_frame["source_tasks"].map(
        lambda values: ",".join(values)
    )
    csv_frame.insert(3, "result_release_id", config["release_id"])
    csv_frame.to_csv(CSV_PATH, index=False, float_format="%.8g")

    print(
        "Built leaderboard: "
        f"{len(payload['rankings']['overall'])} overall-ranked models, "
        f"{len(payload['datasets'])} datasets, "
        f"{len(payload['results'])} result units."
    )


if __name__ == "__main__":
    main()
