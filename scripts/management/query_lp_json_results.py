"""Interactive query tool for LP results stored as JSON files (mRNABench v1.3).

Searches across ALL lp_results/ directories under a root data directory and
prints rows that match the given filters.  All filters are optional and
AND-combined.  String filters use fnmatch-style substring matching, so ``*``
is a wildcard (e.g. ``--model orthrus*`` matches every Orthrus variant).

Multiple values can be passed to any filter; include filters are OR-combined
within the same flag, exclude filters are AND-combined (exclude all of them).

Filename format (v1.3):
  result_lp_{dataset}_{model}_{task}_tcol-{target_col}_split-{split_type}_rs-{seed}.json

Examples
--------
# All results for one model across every dataset
python scripts/management/query_lp_json_results.py --model orthrus-deep-4

# Multiple models at once
python scripts/management/query_lp_json_results.py --model orthrus-deep-4 rna-fm

# All results for a model on a specific task
python scripts/management/query_lp_json_results.py --model orthrus-deep-4 --task reg_ridge

# Filter by dataset and split
python scripts/management/query_lp_json_results.py --dataset prot-loc --split-type homology

# Exclude multiple datasets
python scripts/management/query_lp_json_results.py --model rna-fm --exclude-dataset rnahl eclip

# Show metric values inline
python scripts/management/query_lp_json_results.py --model rna-fm --metrics

# Count matching rows only
python scripts/management/query_lp_json_results.py --model rna-fm --count

# Delete all files for a deprecated model (confirms before deleting)
python scripts/management/query_lp_json_results.py --model old-model-name --delete

# Find all incomplete (model, task, target_col, split_type) combos
python scripts/management/query_lp_json_results.py --incomplete

# Check incomplete combos for a specific dataset or model
python scripts/management/query_lp_json_results.py --incomplete --dataset mrl-sample --model naive

# Check incomplete combos against a custom seed set
python scripts/management/query_lp_json_results.py --incomplete --expected-seeds 2541 413 411

# Rename a model across all datasets (renames JSON files in-place)
python scripts/management/query_lp_json_results.py --rename-model GENERanno-v1 generanno-v1

# Dry-run a rename
python scripts/management/query_lp_json_results.py --rename-model GENERanno-v1 generanno-v1 --dry-run
"""

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

def _default_data_dir() -> str:
    try:
        from mrna_bench.utils import get_data_path
        return get_data_path()
    except Exception:
        return ""

_DEFAULT_SEEDS = ["2541", "413", "411", "412", "2547", "321", "421", "311", "2516", "2515"]

_COLUMNS = ("dataset", "model", "task", "target_col", "split_type", "seed")
_INCOMPLETE_COLUMNS = ("dataset", "model", "task", "target_col", "split_type", "have", "missing")

_TASKS = ("reg_ridge", "reg_lin", "regression", "classification", "multilabel", "zeroshot")

_FILENAME_RE = re.compile(
    r"^result_lp_"
    r"(?P<dataset>[^_]+)"
    r"_(?P<model>.+)"
    r"_(?P<task>reg_ridge|reg_lin|regression|classification|multilabel|zeroshot)"
    r"_tcol-(?P<target_col>.+)"
    r"_split-(?P<split_type>[^_]+)"
    r"_rs-(?P<seed>[^_]+)"
    r"\.json$"
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

class LPRecord(NamedTuple):
    dataset: str
    model: str
    task: str
    target_col: str
    split_type: str
    seed: str
    path: Path


def _parse_filename(path: Path) -> LPRecord | None:
    m = _FILENAME_RE.match(path.name)
    if m is None:
        return None
    return LPRecord(
        dataset=m.group("dataset"),
        model=m.group("model"),
        task=m.group("task"),
        target_col=m.group("target_col"),
        split_type=m.group("split_type"),
        seed=m.group("seed"),
        path=path,
    )


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _match(value: str, pattern: str) -> bool:
    return fnmatch.fnmatch(value.lower(), pattern.lower())


def _auto_pattern(val: str, exact: bool) -> str:
    if exact or "*" in val or "?" in val:
        return val
    return f"*{val}*"


def _matches_any(value: str, patterns: list[str], exact: bool) -> bool:
    return any(_match(value, _auto_pattern(p, exact)) for p in patterns)


def _record_passes(rec: LPRecord, args) -> bool:
    exact: bool = args.exact

    for field, includes, excludes in [
        (rec.model,      args.model,       args.exclude_model),
        (rec.task,       args.task,        args.exclude_task),
        (rec.target_col, args.target_col,  args.exclude_target_col),
        (rec.split_type, args.split_type,  args.exclude_split_type),
        (rec.seed,       args.seed,        args.exclude_seed),
    ]:
        if includes and not _matches_any(field, includes, exact):
            return False
        if excludes and _matches_any(field, excludes, exact):
            return False
    return True


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _iter_datasets(
    data_dir: Path,
    dataset_filters: list[str] | None,
    dataset_excludes: list[str] | None,
    exact: bool = False,
):
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        lp_dir = d / "lp_results"
        if not lp_dir.exists():
            continue
        if dataset_filters and not _matches_any(d.name, dataset_filters, exact):
            continue
        if dataset_excludes and _matches_any(d.name, dataset_excludes, exact):
            continue
        yield d.name, lp_dir


def _collect_records(data_dir: Path, args) -> list[LPRecord]:
    records: list[LPRecord] = []
    for _dataset_name, lp_dir in _iter_datasets(
        data_dir, args.dataset, args.exclude_dataset, args.exact
    ):
        for f in sorted(lp_dir.iterdir()):
            rec = _parse_filename(f)
            if rec is None:
                continue
            if _record_passes(rec, args):
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_metrics(path: Path, all_splits: bool = False) -> str:
    try:
        d: dict = json.loads(path.read_text())
        if not all_splits:
            d = {k: v for k, v in d.items() if k.startswith("test_")}
        return "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in d.items()
        )
    except Exception as exc:
        return f"<error reading metrics: {exc}>"


def _col_widths(records: list[LPRecord]) -> list[int]:
    widths = [len(h) for h in _COLUMNS]
    for rec in records:
        for i, val in enumerate((rec.dataset, rec.model, rec.task,
                                  rec.target_col, rec.split_type, rec.seed)):
            widths[i] = max(widths[i], len(val))
    return widths


def _print_table(records: list[LPRecord], show_metrics: bool, all_splits: bool = False) -> None:
    if not records:
        print("(no results)")
        return

    widths = _col_widths(records)
    header = "  ".join(h.upper().ljust(w) for h, w in zip(_COLUMNS, widths))
    sep = "  ".join("-" * w for w in widths)
    print(header)
    print(sep)

    for rec in records:
        row = (rec.dataset, rec.model, rec.task,
               rec.target_col, rec.split_type, rec.seed)
        line = "  ".join(str(v).ljust(w) for v, w in zip(row, widths))
        if show_metrics:
            line += "  " + _format_metrics(rec.path, all_splits)
        print(line)


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


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _run_incomplete(data_dir: Path, args) -> None:
    import copy
    args_no_seed = copy.copy(args)
    args_no_seed.seed = None
    args_no_seed.exclude_seed = None
    # Don't filter out zeroshot here — that's handled below
    all_records = _collect_records(data_dir, args_no_seed)

    # Group by (dataset, model, task, target_col, split_type)
    from collections import defaultdict
    groups: dict[tuple, set[str]] = defaultdict(set)
    for rec in all_records:
        if rec.task == "zeroshot":
            continue
        key = (rec.dataset, rec.model, rec.task, rec.target_col, rec.split_type)
        groups[key].add(rec.seed)

    expected = set(args.expected_seeds)
    all_incomplete: list[tuple] = []
    ds_counts: dict[str, int] = {}

    for (dataset, model, task, target_col, split_type), have in sorted(groups.items()):
        missing = sorted(expected - have)
        if missing:
            all_incomplete.append((dataset, model, task, target_col, split_type,
                                   sorted(have), missing))
            ds_counts[dataset] = ds_counts.get(dataset, 0) + 1

    _print_incomplete_table(all_incomplete)
    total = len(all_incomplete)
    print(f"\n{total} incomplete combo(s) across {len(ds_counts)} dataset(s)")
    if ds_counts:
        for ds, n in sorted(ds_counts.items()):
            print(f"  {ds}: {n}")


def _run_rename(data_dir: Path, args, records: list[LPRecord]) -> None:
    old_name, new_name = args.rename_model
    dry_run: bool = args.dry_run

    # Exact match on the model field
    to_rename = [r for r in records if r.model == old_name]
    if not to_rename:
        print(f"No files found with model='{old_name}' (exact match).")
        return

    ds_with_old = sorted({r.dataset for r in to_rename})
    action = "Would rename" if dry_run else "Renaming"
    print(f"{action} model '{old_name}' → '{new_name}' "
          f"({len(to_rename)} file(s) across {len(ds_with_old)} dataset(s)):")
    for ds in ds_with_old:
        n = sum(1 for r in to_rename if r.dataset == ds)
        print(f"  {ds}: {n} file(s)")

    if not dry_run:
        answer = input("\nType 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return

    renamed = errors = 0
    for rec in to_rename:
        new_filename = rec.path.name.replace(
            f"_{old_name}_", f"_{new_name}_", 1
        )
        new_path = rec.path.parent / new_filename
        if dry_run:
            print(f"  would rename: {rec.path.name}")
            print(f"             → {new_filename}")
            renamed += 1
        else:
            try:
                rec.path.rename(new_path)
                renamed += 1
            except OSError as exc:
                print(f"  ERROR: {rec.path}: {exc}", file=sys.stderr)
                errors += 1

    print()
    label = "Dry-run summary (no changes made):" if dry_run else "Summary:"
    print(label)
    print(f"  Files {'would be ' if dry_run else ''}renamed: {renamed}")
    if errors:
        print(f"  Errors: {errors}", file=sys.stderr)


def _run_delete(records: list[LPRecord]) -> None:
    total = len(records)
    ds_counts: dict[str, int] = {}
    for rec in records:
        ds_counts[rec.dataset] = ds_counts.get(rec.dataset, 0) + 1

    print(f"\nAbout to DELETE {total} file(s) across {len(ds_counts)} dataset(s). "
          "This cannot be undone.")
    answer = input("Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return

    deleted = errors = 0
    for rec in records:
        try:
            rec.path.unlink()
            deleted += 1
        except OSError as exc:
            print(f"  ERROR deleting {rec.path}: {exc}", file=sys.stderr)
            errors += 1

    print(f"\nDeleted {deleted} file(s).")
    if errors:
        print(f"Errors: {errors}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query LP results stored as JSON files (mRNABench v1.3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory (default: mrna_bench configured data path)")

    g = parser.add_argument_group(
        "filters  (all optional; multiple values are OR-combined within a flag; * = wildcard)"
    )
    g.add_argument("--dataset",    metavar="PATTERN", nargs="+")
    g.add_argument("--model",      metavar="PATTERN", nargs="+")
    g.add_argument("--task",       metavar="PATTERN", nargs="+")
    g.add_argument("--target-col", metavar="PATTERN", nargs="+", dest="target_col")
    g.add_argument("--split-type", metavar="PATTERN", nargs="+", dest="split_type")
    g.add_argument("--seed",       metavar="PATTERN", nargs="+")

    e = parser.add_argument_group(
        "exclusion filters  (multiple values are AND-combined — all are excluded)"
    )
    e.add_argument("--exclude-dataset",    metavar="PATTERN", nargs="+", dest="exclude_dataset")
    e.add_argument("--exclude-model",      metavar="PATTERN", nargs="+", dest="exclude_model")
    e.add_argument("--exclude-task",       metavar="PATTERN", nargs="+", dest="exclude_task")
    e.add_argument("--exclude-target-col", metavar="PATTERN", nargs="+",
                   dest="exclude_target_col")
    e.add_argument("--exclude-split-type", metavar="PATTERN", nargs="+",
                   dest="exclude_split_type")
    e.add_argument("--exclude-seed",       metavar="PATTERN", nargs="+", dest="exclude_seed")

    parser.add_argument("--exact", action="store_true",
                        help="Exact matching instead of substring matching")

    parser.add_argument("--metrics", action="store_true",
                        help="Print test metric values beneath each row")
    parser.add_argument("--all-metrics", action="store_true", dest="all_metrics",
                        help="With --metrics: also show train and val splits")
    parser.add_argument("--count", action="store_true",
                        help="Print row count only, no table")

    parser.add_argument("--delete", action="store_true",
                        help="Delete all matching JSON files (prompts for confirmation)")
    parser.add_argument("--rename-model", metavar=("OLD", "NEW"), nargs=2,
                        dest="rename_model",
                        help="Rename model in filenames across all datasets "
                             "(exact match on OLD name; use --dry-run to preview)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --rename-model: print changes without applying them")
    parser.add_argument("--incomplete", action="store_true",
                        help="Show (model, task, target_col, split_type) combos "
                             "missing seeds from the expected seed set")
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

    if args.incomplete:
        _run_incomplete(data_dir, args)
        return

    records = _collect_records(data_dir, args)
    total = len(records)

    if args.rename_model:
        _run_rename(data_dir, args, records)
        return

    if args.count:
        ds_counts: dict[str, int] = {}
        for rec in records:
            ds_counts[rec.dataset] = ds_counts.get(rec.dataset, 0) + 1
        print(f"{total} file(s) across {len(ds_counts)} dataset(s)")
        for ds, n in sorted(ds_counts.items()):
            print(f"  {ds}: {n}")
        return

    _print_table(records, show_metrics=args.metrics, all_splits=args.all_metrics)
    ds_counts = {}
    for rec in records:
        ds_counts[rec.dataset] = ds_counts.get(rec.dataset, 0) + 1
    print(f"\n{total} file(s) across {len(ds_counts)} dataset(s)")

    if not args.delete or total == 0:
        return

    _run_delete(records)


if __name__ == "__main__":
    main()
