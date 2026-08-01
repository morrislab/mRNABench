"""Interactive query tool for LP results stored in SQLite databases.

Searches across ALL results.db files under a root data directory and prints
rows that match the given filters.  All filters are optional and AND-combined.
String filters use SQL LIKE matching, so ``%`` and ``_`` are wildcards
(e.g. ``--model orthrus%`` matches every Orthrus variant).

Multiple values can be passed to any filter; include filters are OR-combined
within the same flag, exclude filters are AND-combined (exclude all of them).

Examples
--------
# All results for one model across every dataset
python scripts/linear_probe/query_lp_db_results.py --model orthrus-deep-4

# Multiple models at once
python scripts/linear_probe/query_lp_db_results.py --model orthrus-deep-4 rna-fm

# All results for a model on a specific task
python scripts/linear_probe/query_lp_db_results.py --model orthrus-deep-4 --task reg_ridge

# Filter by dataset and split
python scripts/linear_probe/query_lp_db_results.py --dataset prot-loc --split-type homology

# Exclude multiple datasets
python scripts/linear_probe/query_lp_db_results.py --model rna-fm --exclude-dataset rnahl eclip

# Show metric values inline
python scripts/linear_probe/query_lp_db_results.py --model rna-fm --metrics

# Count matching rows only
python scripts/linear_probe/query_lp_db_results.py --model rna-fm --count

# Delete all rows for a deprecated model (confirms before deleting)
python scripts/linear_probe/query_lp_db_results.py --model old-model-name --delete

# Find all incomplete (model, task, target_col, split_type) combos (missing from default seed set)
python scripts/linear_probe/query_lp_db_results.py --incomplete

# Check incomplete combos for a specific dataset or model
python scripts/linear_probe/query_lp_db_results.py --incomplete --dataset mrl-sample --model naive

# Check incomplete combos against a custom seed set
python scripts/linear_probe/query_lp_db_results.py --incomplete --expected-seeds 2541 413 411
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

def _default_data_dir() -> str:
    try:
        from mrna_bench.utils import get_data_path
        return get_data_path()
    except Exception:
        return ""

_DEFAULT_SEEDS = ["2541", "413", "411", "412", "2547", "321", "421", "311", "2516", "2515"]

_COLUMNS = ("dataset", "model", "task", "target_col", "split_type", "seed")
_INCOMPLETE_COLUMNS = ("dataset", "model", "task", "target_col", "split_type", "have", "missing")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _iter_databases(
    data_dir: Path,
    dataset_filters: list[str] | None,
    dataset_excludes: list[str] | None,
    exact: bool = False,
):
    """Yield (dataset_name, db_path) for every results.db found."""
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        db_path = d / "results.db"
        if not db_path.exists():
            continue
        if dataset_filters and not any(_like(d.name, _auto_pattern(f, exact)) for f in dataset_filters):
            continue
        if dataset_excludes and any(_like(d.name, _auto_pattern(f, exact)) for f in dataset_excludes):
            continue
        yield d.name, db_path


def _like(value: str, pattern: str) -> bool:
    """Python-side LIKE check (used only for dataset filtering)."""
    import fnmatch
    return fnmatch.fnmatch(value.lower(), pattern.lower().replace("%", "*").replace("_", "?"))


def _auto_pattern(val: str, exact: bool = False) -> str:
    """Wrap val with %…% for substring matching unless exact or already contains %."""
    if exact or "%" in val:
        return val
    return f"%{val}%"


def _or_clause(col: str, vals: list[str], exact: bool) -> tuple[str, list[str]]:
    """Build ``(col LIKE ? OR col LIKE ? …)`` for multiple include values."""
    placeholders = " OR ".join(f"{col} LIKE ?" for _ in vals)
    return f"({placeholders})", [_auto_pattern(v, exact) for v in vals]


def _and_not_clause(col: str, vals: list[str], exact: bool) -> tuple[str, list[str]]:
    """Build ``col NOT LIKE ? AND col NOT LIKE ? …`` for multiple exclude values."""
    placeholders = " AND ".join(f"{col} NOT LIKE ?" for _ in vals)
    return placeholders, [_auto_pattern(v, exact) for v in vals]


def _build_where(args) -> tuple[str, list]:
    """Build a WHERE clause and parameter list from CLI args."""
    clauses: list[str] = []
    params: list[str] = []
    exact: bool = args.exact

    for col, vals, excls in [
        ("model",      args.model,       args.exclude_model),
        ("task",       args.task,        args.exclude_task),
        ("target_col", args.target_col,  args.exclude_target_col),
        ("split_type", args.split_type,  args.exclude_split_type),
        ("seed",       args.seed,        args.exclude_seed),
    ]:
        if vals:
            clause, p = _or_clause(col, vals, exact)
            clauses.append(clause)
            params.extend(p)
        if excls:
            clause, p = _and_not_clause(col, excls, exact)
            clauses.append(clause)
            params.extend(p)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _query(db_path: Path, where: str, params: list) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT model, task, target_col, split_type, seed, metrics FROM lp_results {where}",
        params,
    ).fetchall()
    conn.close()
    return rows


def _rename_model(db_path: Path, old_name: str, new_name: str) -> int:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.execute(
        "UPDATE lp_results SET model=? WHERE model=?", (new_name, old_name)
    )
    n = cur.rowcount
    conn.commit()
    conn.close()
    return n


def _query_incomplete(
    db_path: Path,
    extra_where: str,
    extra_params: list,
    expected_seeds: set[str],
) -> list[tuple]:
    """Return (model, task, target_col, split_type, have, missing) for combos
    that are missing at least one seed from expected_seeds.

    extra_where/extra_params come from _build_where but with seed filters
    stripped (seed completeness is the whole point here).
    """
    clauses = ["task != 'zeroshot'"]
    params: list = []
    if extra_where:
        clauses.append(extra_where[len("WHERE "):])
        params.extend(extra_params)

    where = "WHERE " + " AND ".join(clauses)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT model, task, target_col, split_type, GROUP_CONCAT(seed) AS seeds"
        f" FROM lp_results {where}"
        f" GROUP BY model, task, target_col, split_type",
        params,
    ).fetchall()
    conn.close()

    incomplete = []
    for r in rows:
        have = set(r["seeds"].split(","))
        missing = sorted(expected_seeds - have)
        if missing:
            incomplete.append((
                r["model"], r["task"], r["target_col"],
                r["split_type"], sorted(have), missing,
            ))
    return incomplete


def _delete(db_path: Path, where: str, params: list) -> int:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.execute(
        f"DELETE FROM lp_results {where}", params
    )
    n = cur.rowcount
    conn.commit()
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_incomplete_table(
    all_rows: list[tuple],  # (dataset, model, task, target_col, split_type, have, missing)
) -> None:
    if not all_rows:
        print("(all combos complete)")
        return

    cols = _INCOMPLETE_COLUMNS
    widths = [len(h) for h in cols]
    for row in all_rows:
        dataset, model, task, target_col, split_type, have, missing = row
        vals = (dataset, model, task, target_col, split_type,
                str(len(have)), str(missing))
        for i, v in enumerate(vals):
            widths[i] = max(widths[i], len(str(v)))

    header = "  ".join(h.upper().ljust(w) for h, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    print(header)
    print(sep)

    for row in all_rows:
        dataset, model, task, target_col, split_type, have, missing = row
        vals = (dataset, model, task, target_col, split_type,
                str(len(have)), str(missing))
        print("  ".join(str(v).ljust(w) for v, w in zip(vals, widths)))


def _format_metrics(metrics_json: str, all_splits: bool = False) -> str:
    d: dict = json.loads(metrics_json)
    if not all_splits:
        d = {k: v for k, v in d.items() if k.startswith("test_")}
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in d.items())


def _col_widths(rows_with_dataset: list[tuple]) -> list[int]:
    widths = [len(h) for h in _COLUMNS]
    for row in rows_with_dataset:
        for i, val in enumerate(row[:len(_COLUMNS)]):
            widths[i] = max(widths[i], len(str(val)))
    return widths


def _print_table(
    all_rows: list[tuple],   # (dataset, model, task, target_col, split_type, seed, metrics_json)
    show_metrics: bool,
    all_splits: bool = False,
) -> None:
    if not all_rows:
        print("(no results)")
        return

    widths = _col_widths(all_rows)
    header = "  ".join(h.upper().ljust(w) for h, w in zip(_COLUMNS, widths))
    sep = "  ".join("-" * w for w in widths)
    print(header)
    print(sep)

    for row in all_rows:
        dataset, model, task, target_col, split_type, seed, metrics_json = row
        line = "  ".join(
            str(v).ljust(w)
            for v, w in zip((dataset, model, task, target_col, split_type, seed), widths)
        )
        if show_metrics:
            line += "  " + _format_metrics(metrics_json, all_splits)
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query LP results across all results.db files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory (default: mrna_bench configured data path)")

    # Filters (nargs="+" = one or more values, OR-combined)
    g = parser.add_argument_group(
        "filters  (all optional; multiple values are OR-combined within a flag; %% = wildcard)"
    )
    g.add_argument("--dataset",    metavar="PATTERN", nargs="+", help="Dataset name(s)")
    g.add_argument("--model",      metavar="PATTERN", nargs="+", help="Model short name(s)")
    g.add_argument("--task",       metavar="PATTERN", nargs="+", help="Task type(s)")
    g.add_argument("--target-col", metavar="PATTERN", nargs="+", dest="target_col",
                   help="Target column(s)")
    g.add_argument("--split-type", metavar="PATTERN", nargs="+", dest="split_type",
                   help="Split type(s)")
    g.add_argument("--seed",       metavar="PATTERN", nargs="+", help="Random seed(s) or 'all'")

    # Exclusion filters (NOT LIKE; multiple values all excluded)
    e = parser.add_argument_group(
        "exclusion filters  (multiple values are AND-combined — all are excluded)"
    )
    e.add_argument("--exclude-dataset",    metavar="PATTERN", nargs="+", dest="exclude_dataset",
                   help="Exclude dataset(s) matching these patterns")
    e.add_argument("--exclude-model",      metavar="PATTERN", nargs="+", dest="exclude_model",
                   help="Exclude model(s) matching these patterns")
    e.add_argument("--exclude-task",       metavar="PATTERN", nargs="+", dest="exclude_task",
                   help="Exclude task(s) matching these patterns")
    e.add_argument("--exclude-target-col", metavar="PATTERN", nargs="+",
                   dest="exclude_target_col",
                   help="Exclude target column(s) matching these patterns")
    e.add_argument("--exclude-split-type", metavar="PATTERN", nargs="+",
                   dest="exclude_split_type",
                   help="Exclude split type(s) matching these patterns")
    e.add_argument("--exclude-seed",       metavar="PATTERN", nargs="+", dest="exclude_seed",
                   help="Exclude seed(s) matching these patterns")

    # Matching mode
    parser.add_argument("--exact", action="store_true",
                        help="Use exact matching instead of substring matching for all filters")

    # Output options
    parser.add_argument("--metrics", action="store_true",
                        help="Print test metric values beneath each row")
    parser.add_argument("--all-metrics", action="store_true", dest="all_metrics",
                        help="With --metrics: also show train and val splits")
    parser.add_argument("--count", action="store_true",
                        help="Print row count only, no table")

    # Actions
    parser.add_argument("--delete", action="store_true",
                        help="Delete all matching rows (prompts for confirmation)")
    parser.add_argument("--rename-model", metavar=("OLD", "NEW"), nargs=2,
                        dest="rename_model",
                        help="Rename a model across all datasets (exact match on OLD name)")
    parser.add_argument("--incomplete", action="store_true",
                        help="Show combos missing seeds from the expected seed set")
    parser.add_argument("--expected-seeds", metavar="SEED", nargs="+", dest="expected_seeds",
                        default=_DEFAULT_SEEDS,
                        help=f"Expected seed set (default: {_DEFAULT_SEEDS})")

    args = parser.parse_args()

    raw = args.data_dir or _default_data_dir()
    if not raw:
        sys.exit("ERROR: --data-dir not specified and mrna_bench data path is not configured.")
    data_dir = Path(raw)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data directory does not exist: {data_dir}")

    # Build WHERE clause; for --incomplete, strip seed filters (irrelevant)
    if args.incomplete:
        import copy
        args_no_seed = copy.copy(args)
        args_no_seed.seed = None
        args_no_seed.exclude_seed = None
        where, params = _build_where(args_no_seed)

        expected = set(args.expected_seeds)
        all_incomplete: list[tuple] = []
        ds_counts: dict[str, int] = {}
        for dataset_name, db_path in _iter_databases(
            data_dir, args.dataset, args.exclude_dataset, args.exact
        ):
            rows = _query_incomplete(db_path, where, params, expected)
            if rows:
                ds_counts[dataset_name] = len(rows)
                for r in rows:
                    all_incomplete.append((dataset_name,) + r)

        _print_incomplete_table(all_incomplete)
        total = len(all_incomplete)
        print(f"\n{total} incomplete combo(s) across {len(ds_counts)} dataset(s)")
        if ds_counts:
            for ds, n in sorted(ds_counts.items()):
                print(f"  {ds}: {n}")
        return

    where, params = _build_where(args)

    # --- collect results ---
    all_rows: list[tuple] = []
    db_counts: dict[str, int] = {}

    for dataset_name, db_path in _iter_databases(data_dir, args.dataset, args.exclude_dataset, args.exact):
        rows = _query(db_path, where, params)
        if rows:
            db_counts[dataset_name] = len(rows)
            for r in rows:
                all_rows.append((
                    dataset_name,
                    r["model"], r["task"], r["target_col"],
                    r["split_type"], r["seed"],
                    r["metrics"],
                ))

    total = len(all_rows)

    if args.count:
        print(f"{total} row(s) across {len(db_counts)} dataset(s)")
        for ds, n in sorted(db_counts.items()):
            print(f"  {ds}: {n}")
        return

    if args.rename_model:
        old_name, new_name = args.rename_model
        # Count only rows matching the exact old model name across the filtered datasets
        rename_total = sum(1 for r in all_rows if r[1] == old_name)
        if rename_total == 0:
            print(f"No rows found with model='{old_name}' (exact match).")
            return
        ds_with_old = sorted({r[0] for r in all_rows if r[1] == old_name})
        print(f"Renaming model '{old_name}' -> '{new_name}' ({rename_total} row(s) across {len(ds_with_old)} dataset(s)):")
        for ds in ds_with_old:
            n = sum(1 for r in all_rows if r[0] == ds and r[1] == old_name)
            print(f"  {ds}: {n} row(s)")
        answer = input("\nType 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return
        renamed = 0
        for dataset_name, db_path in _iter_databases(data_dir, args.dataset, args.exclude_dataset, args.exact):
            n = _rename_model(db_path, old_name, new_name)
            if n:
                renamed += n
        print(f"Renamed {renamed} row(s) total.")
        return

    _print_table(all_rows, show_metrics=args.metrics, all_splits=args.all_metrics)
    print(f"\n{total} row(s) across {len(db_counts)} dataset(s)")

    if not args.delete or total == 0:
        return

    # --- delete ---
    print(f"\nAbout to DELETE {total} row(s). This cannot be undone.")
    answer = input("Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return

    deleted = 0
    for dataset_name, db_path in _iter_databases(data_dir, args.dataset, args.exclude_dataset, args.exact):
        n = _delete(db_path, where, params)
        if n:
            print(f"  {dataset_name}: deleted {n} row(s)")
            deleted += n
    print(f"\nDeleted {deleted} row(s) total.")


if __name__ == "__main__":
    main()
