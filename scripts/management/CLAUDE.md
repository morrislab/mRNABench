# Management Scripts

Tools for managing the LP results and embedding files stored under the data
root.

## Data layout

```
{data_root}/
└── {dataset}/
    ├── embeddings/          # .npz or .h5 — one file per (dataset, model)
    ├── lp_results/          # v1.3 JSON files (mRNABench ≤ 1.3, now legacy)
    └── results.db           # v2.0 SQLite DB (mRNABench ≥ 2.0, current)
```

## Files

### `query_lp_db_results.py` — query v2.0 SQLite results (current)

Searches all `results.db` files. All filters optional, substring-matched by
default, OR-combined within a flag.

```bash
# Query / display
python query_lp_db_results.py --model orthrus --dataset mrl-hl-lbkwk
python query_lp_db_results.py --model rna-fm --metrics          # test metrics inline
python query_lp_db_results.py --model rna-fm --metrics --all-metrics  # + train/val
python query_lp_db_results.py --model rna-fm --count
python query_lp_db_results.py --incomplete                      # missing seeds

# Mutating actions (both prompt for confirmation)
python query_lp_db_results.py --model old-name --delete
python query_lp_db_results.py --rename-model OLD NEW            # exact match on OLD
```

Key flags: `--dataset`, `--model`, `--task`, `--target-col`, `--split-type`,
`--seed`, `--exclude-*` variants, `--exact`, `--expected-seeds`.

### `query_lp_json_results.py` — query v1.3 JSON results (legacy)

Same interface as `query_lp_db_results.py` but reads filenames under
`lp_results/` instead of a database.  Filename format:

```
result_lp_{dataset}_{model}_{task}_tcol-{target_col}_split-{split_type}_rs-{seed}.json
```

`--rename-model` renames the JSON files on disk; supports `--dry-run` to
preview without applying.  Use this (not `query_lp_db_results.py`) for any
data that was produced by mRNABench ≤ 1.3.

### `manage_embeddings.py` — manage embedding files

```bash
python manage_embeddings.py --model splicebert           # list
python manage_embeddings.py --model splicebert --stats   # disk usage
python manage_embeddings.py --missing                    # in DB but no .npz/.h5
python manage_embeddings.py --model old --delete
python manage_embeddings.py --rename-model OLD NEW       # exact match on OLD
```

`--rename-model` always does an exact match on OLD regardless of `--exact`.

### `migrate_json_to_sqlite.py` — one-time v1.3 → v2.0 migration

Reads all `lp_results/*.json` files and upserts them into `results.db`.
Old JSON files are not deleted.  Safe to re-run (uses `INSERT OR REPLACE`).

```bash
python migrate_json_to_sqlite.py --dry-run      # parse only, no writes
python migrate_json_to_sqlite.py                # migrate
python migrate_json_to_sqlite.py --verify-only  # check DB matches JSON files
```

## Renaming a model (both data stores)

When a model's short name changes you need to update four places:

1. The `get_model_short_name()` method in `mrna_bench/models/<model>.py`
2. JSON result files (v1.3): `query_lp_json_results.py --rename-model OLD NEW`
3. SQLite DB rows (v2.0): `query_lp_db_results.py --rename-model OLD NEW`
4. Embedding files: `manage_embeddings.py --rename-model OLD NEW`

If the old and new names are a direct swap of two existing names, use a
3-step rename through a temporary name to avoid collisions:

```bash
# Step 1 — move A out of the way
python <script>.py --rename-model A A-SWAP
# Step 2 — rename B → A
python <script>.py --rename-model B A
# Step 3 — rename temp → B
python <script>.py --rename-model A-SWAP B
```

Repeat all three steps for each of the three `--rename-model` scripts above
(items 2–4).
