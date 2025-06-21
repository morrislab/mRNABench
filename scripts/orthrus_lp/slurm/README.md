# Orthrus Linear-Probe Pipeline – Slurm helpers

This folder contains scripts for running **linear probing** on datasets using
pre-computed Orthrus embeddings.

## 1. Key scripts

| Script | Role |
|--------|------|
| `probe_all_orthrus.py` | Orchestrates Slurm job submission for a grid of `(dataset, target_column, split_type, model_checkpoint)` combinations. |
| `modelversion_slurm.sh` | Lightweight batch wrapper executed on the cluster; activates the Conda environment and calls the worker. |
| `../by_modelversion.py` | Worker script that actually trains / evaluates the linear probe for one seed and persists the metrics. |

## 2. Selecting checkpoints

`probe_all_orthrus.py` now supports **two alternative workflows**:

### A. Automatic discovery (legacy default)
```
python probe_all_orthrus.py \
    --model_dir /path/to/models \
    --best_only            # or --best_onward / --filter_substr SUBSTR
```
• Scans every model directory under `--model_dir`.
• Chooses checkpoints via `get_ckpts_from_dir()` (same logic as the embedding
  pipeline).

### B. Explicit list via config file (new)
```
python probe_all_orthrus.py \
    --model_dir   /path/to/models \
    --config_file scripts/orthrus_lp/slurm/models_to_probe.json \
    --dry_run
```
`--config_file` points to a JSON (or YAML) file that maps **model_version → list
of checkpoint files**. Example:
```json
{
  "orthru_v1": ["best.ckpt"],
  "orthru_v2": ["epoch=0-step=6000.ckpt", "epoch=0-step=8000.ckpt"]
}
```
When this flag is supplied the discovery flags (`--best_only`, `--best_onward`,
`--filter_substr`) are ignored.

## 3. Common options
* `--canonical_split` Restrict probing to each dataset's canonical split only.
* `--force_recompute` Redo probes even if the JSON metrics already exist.
* `--dry_run` Print `sbatch` commands instead of submitting them.

## 4. Typical workflow
1. **Dry-run first** to confirm which jobs will be launched:
   ```bash
   python probe_all_orthrus.py \
       --model_dir /models \
       --config_file scripts/orthrus_lp/slurm/models_to_probe.json \
       --dry_run
   ```
2. **Submit jobs** once happy:
   ```bash
   python probe_all_orthrus.py \
       --model_dir /models \
       --config_file scripts/orthrus_lp/slurm/models_to_probe.json
   ```
3. Monitor log files `logs/orthrus_lp.*.{out,err}` for progress.

---
Questions? Open an issue or ping the maintainers. 