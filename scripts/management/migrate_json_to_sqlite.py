"""Migrate old JSON linear probe results to the new SQLite format.

Reads all result_lp_*.json files from::

    {source_dir}/{dataset}/lp_results/

and writes them into::

    {dest_dir}/{dataset}/results.db

using the same ``lp_results`` table schema as
``mrna_bench.linear_probe.persister.LinearProbePersister``.

Old JSON files are NOT deleted.  Re-running the script is safe because
inserts use ``INSERT OR REPLACE`` (upsert on the composite primary key).

Usage
-----
# Dry-run: parse filenames, print what would be written, write nothing
python scripts/linear_probe/migrate_json_to_sqlite.py --dry-run

# Migrate then verify
python scripts/linear_probe/migrate_json_to_sqlite.py

# Verify only (no writes)
python scripts/linear_probe/migrate_json_to_sqlite.py --verify-only
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LP_SCHEMA = """\
CREATE TABLE IF NOT EXISTS lp_results (
    model      TEXT NOT NULL,
    task       TEXT NOT NULL,
    target_col TEXT NOT NULL,
    split_type TEXT NOT NULL,
    seed       TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, task, target_col, split_type, seed)
)
"""

# Order matters: check "reg_lin" / "reg_ridge" before "regression" to avoid
# matching "reg" as a prefix of the longer task names.
_TASKS = ["reg_lin", "reg_ridge", "regression", "classification", "multilabel"]

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
# SQLite helpers
# ---------------------------------------------------------------------------

def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute(_LP_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(source_dir: Path, dest_dir: Path, *, dry_run: bool = False) -> bool:
    """Migrate JSON results to SQLite.

    Args:
        source_dir: Root directory whose sub-directories are dataset dirs.
        dest_dir: Root directory where ``{dataset}/results.db`` files are written.
        dry_run: If True, parse filenames and report without writing anything.

    Returns:
        True if migration completed with zero errors.
    """
    total_migrated = 0
    total_errors = 0

    for dataset_dir in sorted(source_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        lp_dir = dataset_dir / "lp_results"
        if not lp_dir.exists():
            continue

        dataset_name = dataset_dir.name
        json_files = sorted(lp_dir.glob("result_lp_*.json"))
        if not json_files:
            continue

        print(f"\n[{dataset_name}]  {len(json_files)} JSON files", flush=True)

        if dry_run:
            # Parse a few filenames to sanity-check
            sample_errors = 0
            for jf in json_files[:5]:
                try:
                    parsed = parse_filename(jf.name, dataset_name)
                    print(f"  ok  {jf.name}")
                    print(f"      -> model={parsed['model']!r}  task={parsed['task']!r}"
                          f"  tcol={parsed['target_col']!r}"
                          f"  split={parsed['split_type']!r}  seed={parsed['seed']!r}")
                except ValueError as exc:
                    print(f"  ERR {jf.name}: {exc}")
                    sample_errors += 1
            if len(json_files) > 5:
                print(f"  ... (showing 5 of {len(json_files)})")
            continue

        dest_dataset_dir = dest_dir / dataset_name
        dest_dataset_dir.mkdir(parents=True, exist_ok=True)
        db_path = dest_dataset_dir / "results.db"

        conn = _open_db(db_path)
        dataset_errors = 0

        for jf in json_files:
            try:
                parsed = parse_filename(jf.name, dataset_name)
                metrics = json.loads(jf.read_text())
                conn.execute(
                    "INSERT OR REPLACE INTO lp_results"
                    " (model, task, target_col, split_type, seed, metrics)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        parsed["model"], parsed["task"], parsed["target_col"],
                        parsed["split_type"], parsed["seed"],
                        json.dumps(metrics, default=float),
                    ),
                )
                total_migrated += 1
            except Exception as exc:
                print(f"  ERROR {jf.name}: {exc}", file=sys.stderr)
                dataset_errors += 1
                total_errors += 1

        conn.commit()
        row_count = conn.execute("SELECT COUNT(*) FROM lp_results").fetchone()[0]
        conn.close()

        status = "OK" if dataset_errors == 0 else f"ERRORS={dataset_errors}"
        print(f"  [{status}]  {row_count} rows written to results.db", flush=True)

    print()
    if dry_run:
        print("Dry-run complete — no files were written.")
    else:
        print(f"Migration complete: {total_migrated} rows inserted, {total_errors} errors.")

    return total_errors == 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(source_dir: Path, dest_dir: Path) -> bool:
    """Verify that every JSON result file is present in the SQLite database.

    Args:
        source_dir: Root directory containing dataset sub-directories.
        dest_dir: Root directory containing ``{dataset}/results.db`` files.

    Returns:
        True if all files are present and metric keys match.
    """
    total_files = 0
    total_verified = 0
    total_missing = 0
    total_key_mismatch = 0
    total_parse_errors = 0

    for dataset_dir in sorted(source_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        lp_dir = dataset_dir / "lp_results"
        if not lp_dir.exists():
            continue

        dataset_name = dataset_dir.name
        json_files = sorted(lp_dir.glob("result_lp_*.json"))
        if not json_files:
            continue

        total_files += len(json_files)

        db_path = (dest_dir / dataset_name) / "results.db"
        if not db_path.exists():
            print(f"[MISSING DB]  {dataset_name}  — results.db not found at {db_path}")
            total_missing += len(json_files)
            continue

        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row

        ds_verified = 0
        ds_missing = 0
        ds_key_mismatch = 0
        ds_parse_errors = 0

        for jf in json_files:
            try:
                parsed = parse_filename(jf.name, dataset_name)
            except ValueError as exc:
                print(f"  PARSE ERROR {jf.name}: {exc}", file=sys.stderr)
                ds_parse_errors += 1
                total_parse_errors += 1
                continue

            row = conn.execute(
                "SELECT metrics FROM lp_results"
                " WHERE model=? AND task=? AND target_col=? AND split_type=? AND seed=?",
                (parsed["model"], parsed["task"], parsed["target_col"],
                    parsed["split_type"], parsed["seed"]),
            ).fetchone()

            if row is None:
                ds_missing += 1
                total_missing += 1
                if ds_missing <= 3:
                    print(f"  MISSING  {jf.name}")
            else:
                original_keys = set(json.loads(jf.read_text()).keys())
                db_keys = set(json.loads(row["metrics"]).keys())
                if original_keys != db_keys:
                    ds_key_mismatch += 1
                    total_key_mismatch += 1
                    if ds_key_mismatch <= 3:
                        print(f"  KEY MISMATCH  {jf.name}")
                        print(f"    original: {sorted(original_keys)}")
                        print(f"    database: {sorted(db_keys)}")
                else:
                    ds_verified += 1
                    total_verified += 1

        conn.close()

        issues = ds_missing + ds_key_mismatch + ds_parse_errors
        status = "OK" if issues == 0 else f"ISSUES={issues}"
        print(
            f"[{status}]  {dataset_name}:  "
            f"{len(json_files)} files  |  "
            f"{ds_verified} verified  |  "
            f"{ds_missing} missing  |  "
            f"{ds_key_mismatch} key-mismatch  |  "
            f"{ds_parse_errors} parse-errors",
            flush=True,
        )

    print()
    print(f"TOTAL:  {total_files} files  |  {total_verified} verified  |  "
          f"{total_missing} missing  |  {total_key_mismatch} key-mismatch  |  "
          f"{total_parse_errors} parse-errors")

    return (total_missing + total_key_mismatch + total_parse_errors) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate old LP JSON results to SQLite (results.db).",
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
    args = parser.parse_args()

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
        ok = verify(source_dir, dest_dir)
    else:
        print(f"Migrating JSON results from {source_dir}  ->  {dest_dir}\n")
        ok = migrate(source_dir, dest_dir, dry_run=args.dry_run)

        if not args.dry_run:
            print("\n--- VERIFICATION ---\n")
            ok = verify(source_dir, dest_dir) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
