"""Migrate JSON linear probe results of mRNABench <= 1.3 to results.db.

Reads all result_lp_*.json files from::

    {source_dir}/{dataset}/lp_results/

and writes them into the ``runs`` and ``metrics`` tables of::

    {dest_dir}/{dataset}/results.db

through ``mrna_bench.persister.ResultsPersister``. Each result is stored as a
``linear_probe`` run whose probe config is the estimator those versions used
(RidgeCV with alphas 1e-3 to 10 on raw embeddings for ridge regression, OLS
for ``reg_lin``, and LogisticRegression with C=1 for classification), so
legacy results never share a key with results of the current probe. The
embedding config is mean pooling of the final layer with the model's
default attention implementation.

Old JSON files are NOT deleted. Re-running the script is safe: a run with
the same key is replaced.

Usage
-----
# Dry-run: parse filenames, print what would be written, write nothing
python scripts/management/migrate_json_to_sqlite.py --dry-run

# Migrate then verify
python scripts/management/migrate_json_to_sqlite.py

# Local checkpoints not in the model catalog take a catalog model's defaults
python scripts/management/migrate_json_to_sqlite.py --fallback-model Orthrus

# Verify only (no writes)
python scripts/management/migrate_json_to_sqlite.py --verify-only
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.multioutput import MultiOutputClassifier

from mrna_bench.embedder import get_config_hash, get_embedding_config
from mrna_bench.models import MODEL_CATALOG
from mrna_bench.persister import ResultsPersister, metrics_by_split

# Order matters: check "reg_lin" / "reg_ridge" before "regression" to avoid
# matching "reg" as a prefix of the longer task names.
_TASKS = ["reg_lin", "reg_ridge", "regression", "classification", "multilabel"]

# Estimators of mRNABench <= 1.3 for each task name in the JSON filenames.
_LEGACY_ESTIMATORS = {
    "regression": RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]),
    "reg_lin": LinearRegression(n_jobs=-1),
    "reg_ridge": RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]),
    "classification": LogisticRegression(max_iter=5000, n_jobs=-1),
    "multilabel": MultiOutputClassifier(
        LogisticRegression(max_iter=5000, n_jobs=-1), n_jobs=-1
    ),
}

def _default_data_dir() -> str:
    try:
        from mrna_bench.utils import get_data_path
        return get_data_path()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filename: str, dataset_name: str) -> dict:
    """Parse a ``result_lp_*.json`` filename into its LP components.

    Args:
        filename: Bare filename (no directory path), e.g.
            ``result_lp_prot-loc_orthrus-deep-4_multilabel_tcol-target_split-homology_rs-311.json``
        dataset_name: Name of the dataset directory (used to strip the prefix).

    Returns:
        Dict with keys ``model``, ``task``, ``target_col``, ``split_type``,
        ``seed`` (all strings).

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    if not filename.endswith(".json"):
        raise ValueError(f"Not a JSON file: {filename!r}")
    stem = filename[:-5]  # strip .json

    if not stem.startswith("result_lp_"):
        raise ValueError(f"Unexpected prefix in: {filename!r}")
    stem = stem[len("result_lp_"):]

    # Strip dataset_name prefix (dataset names use hyphens, not underscores)
    dataset_prefix = dataset_name + "_"
    if not stem.startswith(dataset_prefix):
        raise ValueError(
            f"Filename {filename!r} does not start with dataset {dataset_name!r}"
        )
    remainder = stem[len(dataset_prefix):]
    # remainder = "{model}_{task}_tcol-{target_col}_split-{split_type}_rs-{seed}"

    # Locate the task marker: "_{task}_tcol-"
    model: str | None = None
    task: str | None = None
    after_tcol: str | None = None
    for t in _TASKS:
        marker = f"_{t}_tcol-"
        idx = remainder.find(marker)
        if idx != -1:
            model = remainder[:idx]
            task = t
            after_tcol = remainder[idx + len(marker):]
            break

    if task is None or after_tcol is None:
        raise ValueError(f"Could not identify task in: {filename!r}")

    # after_tcol = "{target_col}_split-{split_type}_rs-{seed}"
    # Use rfind so target_col can safely contain underscores.
    rs_idx = after_tcol.rfind("_rs-")
    if rs_idx == -1:
        raise ValueError(f"Missing '_rs-' in: {filename!r}")
    seed = after_tcol[rs_idx + len("_rs-"):]
    before_rs = after_tcol[:rs_idx]  # "{target_col}_split-{split_type}"

    split_idx = before_rs.rfind("_split-")
    if split_idx == -1:
        raise ValueError(f"Missing '_split-' in: {filename!r}")
    split_type = before_rs[split_idx + len("_split-"):]
    target_col = before_rs[:split_idx]

    if not model:
        raise ValueError(f"Empty model name parsed from: {filename!r}")
    if not target_col:
        raise ValueError(f"Empty target_col parsed from: {filename!r}")

    return {
        "model": model,
        "task": task,
        "target_col": target_col,
        "split_type": split_type,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------------

def run_key(parsed: dict) -> tuple:
    """Key shared by the JSON files of one run config (all seeds)."""
    return (
        parsed["model"], parsed["task"], parsed["target_col"],
        parsed["split_type"],
    )


def make_persister(
    dataset_dir: Path,
    key: tuple,
    model_classes: dict,
    fallback,
) -> ResultsPersister | None:
    """Build the persister for a run config, or None for unknown models."""
    model, task, target_col, split_type = key
    model_class = model_classes.get(model, fallback)
    if model_class is None:
        return None
    probe_config = {
        "estimator": " ".join(repr(_LEGACY_ESTIMATORS[task]).split())
    }
    return ResultsPersister(
        SimpleNamespace(dataset_path=str(dataset_dir)),
        model,
        "linear_probe",
        task if task in ("classification", "multilabel") else "regression",
        target_col,
        split_type,
        get_embedding_config(model_class),
        probe_config,
    )


def group_files(lp_dir: Path, dataset_name: str) -> tuple[dict, list]:
    """Group seed files by run config; return groups and parse errors."""
    groups: dict[tuple, dict[str, Path]] = {}
    errors = []
    for jf in sorted(lp_dir.glob("result_lp_*.json")):
        try:
            parsed = parse_filename(jf.name, dataset_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if parsed["seed"] == "all":
            continue  # aggregate over seeds, recomputed from the seed runs
        groups.setdefault(run_key(parsed), {})[parsed["seed"]] = jf
    return groups, errors


def iter_dataset_dirs(source_dir: Path):
    """Yield (dataset_name, lp_results dir) for each dataset with results."""
    for dataset_dir in sorted(source_dir.iterdir()):
        lp_dir = dataset_dir / "lp_results"
        if dataset_dir.is_dir() and lp_dir.exists():
            yield dataset_dir.name, lp_dir


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(
    source_dir: Path,
    dest_dir: Path,
    model_classes: dict,
    fallback,
    *,
    dry_run: bool = False,
) -> bool:
    """Migrate JSON results to results.db.

    Args:
        source_dir: Root directory whose sub-directories are dataset dirs.
        dest_dir: Root directory where ``{dataset}/results.db`` is written.
        model_classes: Model class for each model short name.
        fallback: Model class for short names not in model_classes, or None
            to skip them.
        dry_run: If True, parse filenames and report without writing.

    Returns:
        True if migration completed with zero errors.
    """
    total_migrated = 0
    total_errors = 0
    skipped_models: set[str] = set()

    for dataset_name, lp_dir in iter_dataset_dirs(source_dir):
        groups, parse_errors = group_files(lp_dir, dataset_name)
        for error in parse_errors:
            print(f"  ERROR {error}", file=sys.stderr)
        total_errors += len(parse_errors)

        n_files = sum(len(seeds) for seeds in groups.values())
        print(f"\n[{dataset_name}]  {n_files} seed files in "
              f"{len(groups)} run configs", flush=True)

        dataset_dir = dest_dir / dataset_name
        if not dry_run:
            dataset_dir.mkdir(parents=True, exist_ok=True)

        for i, (key, seed_files) in enumerate(sorted(groups.items())):
            persister = make_persister(dataset_dir, key, model_classes, fallback)
            if persister is None:
                skipped_models.add(key[0])
                continue
            if dry_run:
                if i < 3:
                    print(f"  {key} seeds={sorted(seed_files)}")
                    print(f"    embedding_config={persister.embedding_config}")
                    print(f"    probe_config={persister.probe_config}")
                continue
            try:
                persister.persist_multirun_results({
                    seed: metrics_by_split(json.loads(path.read_text()))
                    for seed, path in seed_files.items()
                })
                total_migrated += len(seed_files)
            except Exception as exc:
                print(f"  ERROR {key}: {exc}", file=sys.stderr)
                total_errors += 1

    if skipped_models:
        print(f"\nSkipped models not in the catalog (use --fallback-model): "
              f"{sorted(skipped_models)}")
    print()
    if dry_run:
        print("Dry-run complete — no files were written.")
    else:
        print(f"Migration complete: {total_migrated} runs written, "
              f"{total_errors} errors.")

    return total_errors == 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(
    source_dir: Path,
    dest_dir: Path,
    model_classes: dict,
    fallback,
) -> bool:
    """Verify that every migrated JSON result is in results.db.

    Args:
        source_dir: Root directory containing dataset sub-directories.
        dest_dir: Root directory containing ``{dataset}/results.db`` files.
        model_classes: Model class for each model short name.
        fallback: Model class for short names not in model_classes.

    Returns:
        True if all results are present with the same metric names.
    """
    total_verified = 0
    total_missing = 0
    total_mismatch = 0

    for dataset_name, lp_dir in iter_dataset_dirs(source_dir):
        groups, _ = group_files(lp_dir, dataset_name)
        stored = {
            (r["model"], r["task"], r["target_col"], r["split_type"],
             r["seed"], r["probe_hash"]): {
                f"{split}_{metric}"
                for split, values in r["metrics"].items()
                for metric in values
            }
            for r in ResultsPersister.load_all_results(dest_dir / dataset_name)
        }

        verified = missing = mismatch = 0
        for key, seed_files in groups.items():
            persister = make_persister(
                dest_dir / dataset_name, key, model_classes, fallback
            )
            if persister is None:
                continue
            probe_hash = get_config_hash(persister.probe_config)
            for seed, path in seed_files.items():
                names = stored.get((
                    persister.model_short_name, persister.task,
                    persister.target_col, persister.split_type, seed,
                    probe_hash,
                ))
                if names is None:
                    missing += 1
                elif names != set(json.loads(path.read_text())):
                    mismatch += 1
                else:
                    verified += 1

        status = "OK" if missing + mismatch == 0 else "ISSUES"
        print(f"[{status}]  {dataset_name}:  {verified} verified  |  "
              f"{missing} missing  |  {mismatch} metric mismatch", flush=True)
        total_verified += verified
        total_missing += missing
        total_mismatch += mismatch

    print(f"\nTOTAL:  {total_verified} verified  |  {total_missing} missing  "
          f"|  {total_mismatch} metric mismatch")
    return total_missing + total_mismatch == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate LP JSON results of mRNABench <= 1.3 to results.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Root directory containing dataset sub-directories with lp_results/ "
             "(default: mrna_bench configured data path)",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Root directory for output results.db files. "
             "Defaults to --source-dir so results.db lands next to lp_results/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse filenames and report what would be written; make no changes.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip migration; only verify existing results.db files.",
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Catalog model whose defaults apply to model names not in the "
             "catalog, e.g. Orthrus for local Orthrus checkpoints.",
    )
    args = parser.parse_args()

    model_classes = {
        cls.get_model_short_name(version): cls
        for cls in MODEL_CATALOG.values()
        for version in cls.valid_versions
    }
    fallback = (
        MODEL_CATALOG[args.fallback_model] if args.fallback_model else None
    )

    raw = args.source_dir or _default_data_dir()
    if not raw:
        sys.exit("ERROR: --source-dir not specified and mrna_bench data path is not configured.")
    source_dir = Path(raw)
    dest_dir = Path(args.dest_dir) if args.dest_dir else source_dir

    if not source_dir.is_dir():
        sys.exit(f"ERROR: source directory does not exist: {source_dir}")

    ok = True

    if args.verify_only:
        print(f"Verifying results in {dest_dir} against JSON files in {source_dir}\n")
        ok = verify(source_dir, dest_dir, model_classes, fallback)
    else:
        print(f"Migrating JSON results from {source_dir}  ->  {dest_dir}\n")
        ok = migrate(
            source_dir, dest_dir, model_classes, fallback,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            print("\n--- VERIFICATION ---\n")
            ok = verify(source_dir, dest_dir, model_classes, fallback) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
