from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

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
}

DATASET_LABELS = {
    "eclip-binding-hepg2": "eCLIP RBP binding (HepG2)",
    "eclip-binding-k562": "eCLIP RBP binding (K562)",
    "mirna-target": "miRNA target binding",
    "mrl-hl-lbkwk": "Paired MRL and RNA half-life",
    "mrl-sample-designed": "MRL MPRA (designed)",
    "mrl-sample-egfp": "MRL MPRA (eGFP)",
    "mrl-sample-mcherry": "MRL MPRA (mCherry)",
    "mrl-sample-varying": "MRL MPRA (varying conditions)",
    "mrl-sugimoto": "Mean ribosome load",
    "rna-lifecycle-ietswaart": "RNA lifecycle",
    "rna-loc-fazal": "RNA subcellular localization",
    "rnahl-human": "RNA half-life (human)",
    "rnahl-mouse": "RNA half-life (mouse)",
    "translation-efficiency-human": "Translation efficiency (human)",
    "translation-efficiency-mouse": "Translation efficiency (mouse)",
    "utr-variants-bohn-utr3": "Curated 3' UTR variants",
    "utr-variants-bohn-utr5": "Curated 5' UTR variants",
    "vep-traitgym-complex": "TraitGym complex-trait variants",
    "vep-traitgym-mendelian": "TraitGym Mendelian variants",
}

TASK_LABELS = {
    "classification": "Classification",
    "multilabel": "Multilabel",
    "regression": "Regression",
}

TASK_ORDER = ("classification", "multilabel", "regression")

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

    excluded_datasets = set(config["excluded_datasets"])
    excluded_models = set(config["excluded_models"])
    frame = frame[~frame["dataset"].isin(excluded_datasets)]
    after_dataset_filter = len(frame)
    frame = frame[~frame["model"].isin(excluded_models)]
    after_model_filter = len(frame)

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
    frame = frame[frame["value"].notna()].copy()
    null_metric_rows = before_null_filter - len(frame)

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
    frame = (
        frame.sort_values(collision_key + ["task_priority"])
        .drop_duplicates(collision_key, keep="first")
        .drop(columns=["task_priority"])
    )
    duplicate_rows = before_null_filter - null_metric_rows - len(frame)

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
        "excluded_rebuilt_dataset_rows": default_split_rows - after_dataset_filter,
        "excluded_model_rows": after_dataset_filter - after_model_filter,
        "excluded_unsupported_task_rows": unsupported_task_rows,
        "excluded_regression_protocol_rows": excluded_regression_protocol_rows,
        "excluded_null_metric_rows": null_metric_rows,
        "resolved_legacy_collision_rows": duplicate_rows,
        "material_legacy_collision_keys": material_collision_count,
        "max_legacy_collision_difference": max_collision_difference,
        "retained_seed_rows": len(frame),
    }
    return frame, stats


def aggregate_units(
    frame: pd.DataFrame,
    expected_runs: int,
    data_status: str,
) -> pd.DataFrame:
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
        frame.groupby(unit_columns, dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs", "std": "std_sample"})
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
    units["coverage_status"] = units["n_runs"].map(
        lambda count: "complete" if count == expected_runs else "incomplete"
    )
    units["expected_runs"] = expected_runs
    units["split_role"] = "test"
    units["higher_is_better"] = True
    units["data_status"] = data_status

    rank_group = [
        "dataset",
        "target",
        "biological_task",
        "result_task",
        "evaluation_method",
        "estimator",
        "split",
        "metric_id",
    ]
    units["unit_rank"] = units.groupby(rank_group)["mean"].rank(
        method="average",
        ascending=False,
    )
    peer_count = units.groupby(rank_group)["model"].transform("count")
    units["unit_percentile"] = (
        1 - (units["unit_rank"] - 1) / (peer_count - 1).clip(lower=1)
    )
    units["peer_count"] = peer_count
    return units


def build_dataset_scores(units: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "model_group",
        "source_group",
        "dataset",
        "biological_task",
        "metric_id",
    ]
    scores = (
        units.groupby(keys)
        .agg(
            score=("unit_percentile", "mean"),
            raw_mean=("mean", "mean"),
            completed_units=("coverage_status", lambda values: (values == "complete").sum()),
            unit_count=("target", "count"),
        )
        .reset_index()
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


def build_task_scores(
    units: pd.DataFrame,
    dataset_scores: pd.DataFrame,
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
    source_scores = (
        dataset_scores.groupby(
            ["model", "model_group", "biological_task", "source_group"]
        )
        .agg(
            score=("score", "mean"),
            dataset_count=("dataset", "count"),
        )
        .reset_index()
    )
    task_scores = (
        source_scores.groupby(["model", "model_group", "biological_task"])
        .agg(
            score=("score", "mean"),
            source_group_count=("source_group", "count"),
            dataset_count=("dataset_count", "sum"),
        )
        .reset_index()
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


def build_consensus(task_scores: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = task_scores[
        task_scores["rank_eligible"]
        & task_scores["biological_task"].isin(TASK_ORDER)
    ]
    rank_matrix = eligible.pivot(index="model", columns="biological_task", values="rank")
    score_matrix = eligible.pivot(index="model", columns="biological_task", values="score")
    rank_matrix = rank_matrix.dropna(subset=list(TASK_ORDER))
    score_matrix = score_matrix.loc[rank_matrix.index]

    families = (
        task_scores[["model", "model_group"]]
        .drop_duplicates("model")
        .set_index("model")["model_group"]
    )
    rows = []
    for model_id in rank_matrix.index:
        task_ranks = {
            task: int(rank_matrix.loc[model_id, task])
            for task in TASK_ORDER
        }
        task_scores_for_model = {
            task: finite(score_matrix.loc[model_id, task])
            for task in TASK_ORDER
        }
        rows.append(
            {
                "model_id": model_id,
                "model_family": families.loc[model_id],
                "rank_sum": sum(task_ranks.values()),
                "worst_task_rank": max(task_ranks.values()),
                "task_ranks": task_ranks,
                "task_scores": task_scores_for_model,
            }
        )

    rows.sort(
        key=lambda row: (
            row["rank_sum"],
            row["worst_task_rank"],
            row["model_id"],
        )
    )
    previous_key = None
    current_rank = 0
    for index, row in enumerate(rows, start=1):
        rank_key = (row["rank_sum"], row["worst_task_rank"])
        if rank_key != previous_key:
            current_rank = index
            previous_key = rank_key
        row["rank"] = current_rank
        row["coverage_ratio"] = 1.0
    return rows


def build_payload(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    normalized, row_stats = normalize_rows(source, config)
    units = aggregate_units(
        normalized,
        expected_runs=int(config["expected_runs"]),
        data_status=str(config["data_status"]),
    )
    dataset_scores = build_dataset_scores(units)
    task_scores = build_task_scores(units, dataset_scores)
    consensus = build_consensus(task_scores)

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
                "ci95_low": finite(row["ci95_low"]),
                "ci95_high": finite(row["ci95_high"]),
                "unit_percentile": finite(row["unit_percentile"]),
                "peer_count": int(row["peer_count"]),
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
        "schema_version": "1.1.0",
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
            "seed_aggregation": "Mean across seeded test results with sample SD and Student-t 95% CI.",
            "regression_result_task": config["regression_result_task"],
            "unit_ranking": "Models are ranked within each dataset-target-metric unit, then converted to a 0-1 percentile.",
            "dataset_weighting": "Targets are averaged within each dataset so multi-target datasets do not dominate.",
            "source_weighting": "Registered dataset IDs are averaged within source dataset groups.",
            "task_weighting": "Source-group percentiles are averaged within each biological task.",
            "consensus": "Sum of classification, multilabel, and regression task ranks for models with complete coverage.",
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
            "consensus": consensus,
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
        f"{len(payload['rankings']['consensus'])} consensus-ranked models, "
        f"{len(payload['datasets'])} datasets, "
        f"{len(payload['results'])} result units."
    )


if __name__ == "__main__":
    main()
