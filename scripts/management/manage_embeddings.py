"""Manage embedding files (.npz / .h5) stored under a data root directory.

Each embedding file lives at::

    {data_dir}/{dataset}/embeddings/{dataset}_{model}.{npz|h5}

All filters are optional and substring-matched by default (use --exact for
exact matching).  Multiple values per flag are OR-combined for include filters
and AND-combined for exclude filters.

Actions
-------
(default)       List matching files in a table
--count         Print match counts only
--stats         Show file sizes and totals
--list-models   Print unique model names found (across matching datasets)
--list-datasets Print unique dataset names that have at least one match
--missing       Show (dataset, model) pairs that have LP results in results.db
                but no corresponding embedding file
--delete        Delete matching files (prompts for confirmation)
--rename-model OLD NEW
                Rename model OLD -> NEW in all matching embedding filenames
                (exact match on OLD regardless of --exact flag)

Examples
--------
# List all embeddings for a model
python scripts/embedding/manage_embeddings.py --model orthrus-deep-4

# List embeddings for multiple models, excluding certain datasets
python scripts/embedding/manage_embeddings.py --model rna-fm ernierna \\
    --exclude-dataset eclip mirna

# Show disk usage by model
python scripts/embedding/manage_embeddings.py --model orthrus --stats

# Find models that have LP results but no embedding file
python scripts/embedding/manage_embeddings.py --missing

# Delete embeddings for a specific model+dataset combo
python scripts/embedding/manage_embeddings.py --model old-model --dataset prot-loc --delete

# Rename a model in all embedding filenames
python scripts/embedding/manage_embeddings.py --rename-model GENERanno-v2-base generanno-v2-base
"""

import argparse
import fnmatch
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
_EMB_EXTS = {".npz", ".h5"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_emb_filename(filename: str, dataset_name: str) -> str | None:
    """Extract model name from ``{dataset}_{model}.{ext}``.

    Returns None if the filename doesn't match the expected pattern.
    """
    stem = Path(filename).stem  # strip extension
    prefix = dataset_name + "_"
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix):]


def _auto_pattern(val: str, exact: bool) -> str:
    if exact or "%" in val:
        return val
    return f"%{val}%"


def _fnmatch_pattern(val: str, exact: bool) -> str:
    """Convert a LIKE-style pattern to fnmatch-style for Python-side matching."""
    if exact:
        return val.lower()
    pat = val.lower()
    if "%" not in pat:
        pat = f"*{pat}*"
    return pat.replace("%", "*")


def _matches(value: str, patterns: list[str], exact: bool) -> bool:
    return any(fnmatch.fnmatch(value.lower(), _fnmatch_pattern(p, exact)) for p in patterns)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_embedding_files(
    data_dir: Path,
    dataset_filters: list[str] | None,
    dataset_excludes: list[str] | None,
    model_filters: list[str] | None,
    model_excludes: list[str] | None,
    exact: bool,
) -> list[tuple[str, str, Path]]:
    """Return list of (dataset_name, model_name, file_path) for matching embeddings."""
    results = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        emb_dir = d / "embeddings"
        if not emb_dir.exists():
            continue
        dataset_name = d.name

        if dataset_filters and not _matches(dataset_name, dataset_filters, exact):
            continue
        if dataset_excludes and _matches(dataset_name, dataset_excludes, exact):
            continue

        for f in sorted(emb_dir.iterdir()):
            if f.suffix not in _EMB_EXTS:
                continue
            model_name = _parse_emb_filename(f.name, dataset_name)
            if model_name is None:
                continue

            if model_filters and not _matches(model_name, model_filters, exact):
                continue
            if model_excludes and _matches(model_name, model_excludes, exact):
                continue

            results.append((dataset_name, model_name, f))
    return results


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def action_list(matches: list[tuple]) -> None:
    if not matches:
        print("(no results)")
        return
    ds_w = max(len("DATASET"), max(len(ds) for ds, _, _ in matches))
    mo_w = max(len("MODEL"), max(len(mo) for _, mo, _ in matches))
    ext_w = max(len("EXT"), max(len(f.suffix) for _, _, f in matches))
    header = f"{'DATASET':<{ds_w}}  {'MODEL':<{mo_w}}  {'EXT':<{ext_w}}"
    print(header)
    print("-" * len(header))
    for ds, mo, f in matches:
        print(f"{ds:<{ds_w}}  {mo:<{mo_w}}  {f.suffix:<{ext_w}}")
    print(f"\n{len(matches)} file(s) across {len({ds for ds, _, _ in matches})} dataset(s)")


def action_count(matches: list[tuple]) -> None:
    from collections import Counter
    counts: Counter = Counter(ds for ds, _, _ in matches)
    total_ds = len(counts)
    print(f"{len(matches)} file(s) across {total_ds} dataset(s)")
    for ds, n in sorted(counts.items()):
        print(f"  {ds}: {n}")


def action_stats(matches: list[tuple]) -> None:
    if not matches:
        print("(no results)")
        return
    ds_w = max(len("DATASET"), max(len(ds) for ds, _, _ in matches))
    mo_w = max(len("MODEL"), max(len(mo) for _, mo, _ in matches))
    sz_w = 10
    header = f"{'DATASET':<{ds_w}}  {'MODEL':<{mo_w}}  {'SIZE':>{sz_w}}"
    print(header)
    print("-" * len(header))

    total = 0
    by_model: dict[str, int] = {}
    for ds, mo, f in matches:
        size = f.stat().st_size
        total += size
        by_model[mo] = by_model.get(mo, 0) + size
        print(f"{ds:<{ds_w}}  {mo:<{mo_w}}  {_fmt_size(size):>{sz_w}}")

    print(f"\nTotal: {_fmt_size(total)} across {len(matches)} file(s)")
    if len(by_model) > 1:
        print("\nBy model:")
        for mo, sz in sorted(by_model.items(), key=lambda x: -x[1]):
            print(f"  {mo}: {_fmt_size(sz)}")


def action_list_models(matches: list[tuple]) -> None:
    models = sorted({mo for _, mo, _ in matches})
    for mo in models:
        print(mo)
    print(f"\n{len(models)} unique model(s)")


def action_list_datasets(matches: list[tuple]) -> None:
    datasets = sorted({ds for ds, _, _ in matches})
    for ds in datasets:
        print(ds)
    print(f"\n{len(datasets)} unique dataset(s)")


def action_missing(
    data_dir: Path,
    dataset_filters: list[str] | None,
    dataset_excludes: list[str] | None,
    model_filters: list[str] | None,
    model_excludes: list[str] | None,
    exact: bool,
) -> None:
    """Print (dataset, model) pairs that have LP results but no embedding file."""
    missing = []

    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        dataset_name = d.name
        db_path = d / "results.db"
        emb_dir = d / "embeddings"

        if dataset_filters and not _matches(dataset_name, dataset_filters, exact):
            continue
        if dataset_excludes and _matches(dataset_name, dataset_excludes, exact):
            continue

        if not db_path.exists():
            continue

        # Models with LP results in the DB
        conn = sqlite3.connect(str(db_path), timeout=30)
        rows = conn.execute("SELECT DISTINCT model FROM lp_results").fetchall()
        conn.close()
        db_models = {r[0] for r in rows}

        # Models with embedding files
        emb_models: set[str] = set()
        if emb_dir.exists():
            for f in emb_dir.iterdir():
                if f.suffix not in _EMB_EXTS:
                    continue
                mo = _parse_emb_filename(f.name, dataset_name)
                if mo:
                    emb_models.add(mo)

        for mo in sorted(db_models - emb_models):
            if model_filters and not _matches(mo, model_filters, exact):
                continue
            if model_excludes and _matches(mo, model_excludes, exact):
                continue
            missing.append((dataset_name, mo))

    if not missing:
        print("No missing embeddings found.")
        return

    ds_w = max(len("DATASET"), max(len(ds) for ds, _ in missing))
    mo_w = max(len("MODEL"), max(len(mo) for _, mo in missing))
    print(f"{'DATASET':<{ds_w}}  {'MODEL':<{mo_w}}")
    print("-" * (ds_w + mo_w + 2))
    for ds, mo in missing:
        print(f"{ds:<{ds_w}}  {mo:<{mo_w}}")
    print(f"\n{len(missing)} missing embedding(s)")


def action_delete(matches: list[tuple]) -> None:
    if not matches:
        print("(no results)")
        return
    print(f"About to DELETE {len(matches)} embedding file(s):")
    for ds, mo, f in matches:
        print(f"  {f}")
    answer = input("\nType 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return
    deleted = 0
    for _, _, f in matches:
        try:
            f.unlink()
            deleted += 1
        except OSError as exc:
            print(f"  ERROR deleting {f}: {exc}", file=sys.stderr)
    print(f"Deleted {deleted} file(s).")


def action_rename_model(
    data_dir: Path,
    old_name: str,
    new_name: str,
    dataset_filters: list[str] | None,
    dataset_excludes: list[str] | None,
    exact: bool,
) -> None:
    # Collect files to rename (always exact match on old_name)
    to_rename: list[tuple[Path, Path]] = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        emb_dir = d / "embeddings"
        if not emb_dir.exists():
            continue
        dataset_name = d.name

        if dataset_filters and not _matches(dataset_name, dataset_filters, exact):
            continue
        if dataset_excludes and _matches(dataset_name, dataset_excludes, exact):
            continue

        for f in sorted(emb_dir.iterdir()):
            if f.suffix not in _EMB_EXTS:
                continue
            mo = _parse_emb_filename(f.name, dataset_name)
            if mo != old_name:
                continue
            new_filename = f"{dataset_name}_{new_name}{f.suffix}"
            to_rename.append((f, f.parent / new_filename))

    if not to_rename:
        print(f"No embedding files found with model='{old_name}' (exact match).")
        return

    print(f"Renaming model '{old_name}' -> '{new_name}' in {len(to_rename)} file(s):")
    for old_path, new_path in to_rename:
        print(f"  {old_path.parent.parent.name}/{old_path.name}")
        print(f"    -> {new_path.name}")
    answer = input("\nType 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return
    renamed = 0
    for old_path, new_path in to_rename:
        try:
            old_path.rename(new_path)
            renamed += 1
        except OSError as exc:
            print(f"  ERROR renaming {old_path}: {exc}", file=sys.stderr)
    print(f"Renamed {renamed} file(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage embedding files across dataset directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory (default: mrna_bench configured data path)")

    g = parser.add_argument_group(
        "filters  (substring match by default; multiple values OR-combined)"
    )
    g.add_argument("--dataset",          metavar="PATTERN", nargs="+", help="Dataset name(s)")
    g.add_argument("--model",            metavar="PATTERN", nargs="+", help="Model name(s)")
    g.add_argument("--exclude-dataset",  metavar="PATTERN", nargs="+", dest="exclude_dataset",
                   help="Exclude dataset(s)")
    g.add_argument("--exclude-model",    metavar="PATTERN", nargs="+", dest="exclude_model",
                   help="Exclude model(s)")
    parser.add_argument("--exact", action="store_true",
                        help="Exact matching instead of substring matching")

    a = parser.add_argument_group("actions  (default: list)")
    a.add_argument("--count",         action="store_true", help="Count matches per dataset")
    a.add_argument("--stats",         action="store_true", help="Show file sizes and totals")
    a.add_argument("--list-models",   action="store_true", dest="list_models",
                   help="Print unique model names found")
    a.add_argument("--list-datasets", action="store_true", dest="list_datasets",
                   help="Print unique dataset names with matches")
    a.add_argument("--missing",       action="store_true",
                   help="Show models with LP results but no embedding file")
    a.add_argument("--delete",        action="store_true",
                   help="Delete matching files (prompts for confirmation)")
    a.add_argument("--rename-model",  metavar=("OLD", "NEW"), nargs=2, dest="rename_model",
                   help="Rename model OLD -> NEW in filenames (exact match on OLD)")

    args = parser.parse_args()

    raw = args.data_dir or _default_data_dir()
    if not raw:
        sys.exit("ERROR: --data-dir not specified and mrna_bench data path is not configured.")
    data_dir = Path(raw)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data directory does not exist: {data_dir}")

    if args.missing:
        action_missing(
            data_dir, args.dataset, args.exclude_dataset,
            args.model, args.exclude_model, args.exact,
        )
        return

    if args.rename_model:
        action_rename_model(
            data_dir, args.rename_model[0], args.rename_model[1],
            args.dataset, args.exclude_dataset, args.exact,
        )
        return

    matches = _iter_embedding_files(
        data_dir, args.dataset, args.exclude_dataset,
        args.model, args.exclude_model, args.exact,
    )

    if args.count:
        action_count(matches)
    elif args.stats:
        action_stats(matches)
    elif args.list_models:
        action_list_models(matches)
    elif args.list_datasets:
        action_list_datasets(matches)
    elif args.delete:
        action_delete(matches)
    else:
        action_list(matches)


if __name__ == "__main__":
    main()
