# Orthrus Embedding Pipeline – Slurm submission helpers

This folder contains the utilities used to create whole–dataset sequence embeddings
with **Orthrus** checkpoints on a Slurm cluster.

## 1. Key scripts

| Script | Role |
|--------|------|
| `all_model_embed.py` | Orchestrates job submission. Decides **which** `(model_version, checkpoint)` pairs still need embeddings and submits one Slurm job per pair. |
| `slurm_script.sh` | Light-weight batch wrapper executed by Slurm. Reserves resources, activates the conda env, and launches the embedding worker. |
| `embed_dataset.py` | The actual worker. Loads the model, runs inference on the dataset, and writes the resulting `.npz` file. |

## 2. Selecting checkpoints

`all_model_embed.py` supports **two mutually-exclusive workflows** for choosing what to embed.

### A. Automatic discovery (default)
```
python all_model_embed.py --model_dir /path/to/models [--best_only | --best_onward] [--filter_substr STR]
```
* Scans every sub-directory in `--model_dir`.
* Uses `get_ckpts_from_dir()` to decide which `.ckpt` files to embed.
  * `--best_only` → Only the top checkpoint per model.
  * `--best_onward` → Top checkpoint **and** every later one.
  * No flag → Every checkpoint except the `best` one (to avoid duplication).
* Optional `--filter_substr` restricts the scan to model directories whose
  name contains the given substring.

### B. Explicit list via config file
```
python all_model_embed.py --model_dir /path/to/models --config_file models_to_embed.json
```
* Provide a small JSON (or YAML) file that maps model version → list of checkpoints:
  ```json
  {
    "orthru_v1": ["best.ckpt"],
    "orthru_v2": ["epoch=0-step=4000.ckpt", "epoch=0-step=6000.ckpt"]
  }
  ```
* When `--config_file` is given, the discovery flags (`--best_only`,
  `--best_onward`, `--filter_substr`) are ignored.

## 3. Common options
* `--force_recompute` Re-embed even if the `.npz` already exists.
* `--dry_run` Print the `sbatch` commands instead of submitting them.

## 4. Typical workflow

Assuming you have prepared a file called `models_to_eval.json` in this folder
that enumerates the exact checkpoint list you want to embed (see the example
already committed to the repo), the most common invocation now looks like this:

1. **Dry-run first** to verify which `sbatch` commands will be issued:
   ```bash
   python all_model_embed.py \
       --model_dir /path/to/model_repository \
       --config_file scripts/orthrus_embed/slurm/models_to_eval.json \
       --dry_run
   ```
2. **Launch the jobs** once you are happy with the printed commands:
   ```bash
   python all_model_embed.py \
       --model_dir /path/to/model_repository \
       --config_file scripts/orthrus_embed/slurm/models_to_eval.json
   ```
3. Monitor progress via the log files written to `./logs/`.

## 5. Adding new models
For small, ad-hoc sets it is often easiest to create a one-off JSON file and
pass it via `--config_file`. Larger exploratory sweeps can still rely on the
automatic discovery flags.

---
Questions? Open an issue or ping the maintainers. 