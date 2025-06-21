import subprocess
import os
import json

import mrna_bench as mb
from pathlib import Path

import argparse
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG
import yaml

def get_step_from_ckpt(ckpt):
    """
    Extract the step number from the checkpoint filename.
    """
    # Example filename: "epoch=0-step=1000.ckpt"
    # Split by '-' and take the last part, then split by '=' and take the last part

    step = ckpt.split('-')[-1].split('=')[-1].split('.')[0]
    return int(step)

def get_ckpts_from_dir(model_dir, best_only=False, best_onward=False):
    """
    Get the list of checkpoints from the model directory.
    """
    ckpt_files = [f for f in Path(model_dir).iterdir() if f.is_file() and f.suffix == '.ckpt' and "-2k" not in f.stem]

    if best_onward:
        # Filter to only include the best checkpoint and all checkpoints after it
        best_ckpt = [f for f in ckpt_files if "best" in f.stem]
        if len(best_ckpt) > 0:
            best_ckpt = best_ckpt[0]
            ckpt_files = [f for f in ckpt_files if get_step_from_ckpt(f.stem) >= get_step_from_ckpt(best_ckpt.stem) and f.stem != best_ckpt.stem]

    if best_only:
        # Filter to only include the top checkpoint which either has the prefix best or if not available, the last checkpoint
        top_files = [f for f in ckpt_files if "best" in f.stem]

        if len(top_files) < 1:
            # If no best checkpoint is found, use the last checkpoint
            top_files = sorted(ckpt_files, key=lambda x: get_step_from_ckpt(x.stem), reverse=True)[:1]

        ckpt_files = top_files

    if not best_onward and not best_only:
        # skip the best checkpoint (it's already included in the full list)
        ckpt_files = [f for f in ckpt_files if "best" not in f.stem]

    return sorted([f.name for f in ckpt_files], key=lambda x: get_step_from_ckpt(x))

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def load_model_ckpt_config(cfg_path):
    """Read a JSON or YAML file that maps model_version -> list[checkpoint].

    The file *must* contain a mapping (dictionary). Each value can be either
    a single string (one checkpoint) or a list of checkpoint strings. All
    checkpoint names must end with ``.ckpt``.

    Parameters
    ----------
    cfg_path : str or Path
        Path to the JSON / YAML configuration file.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from model version to a list of checkpoint filenames.
    """
    path = Path(cfg_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # ------------------------------------------------------------------
    # Choose parser based on file suffix (default to JSON)
    # ------------------------------------------------------------------
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to parse YAML config files. Please `pip install pyyaml`. ")
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of model_version -> list[checkpoint].")

    # Normalise: ensure every value is a list of strings
    clean_data = {}
    for model_version, ckpts in data.items():
        if isinstance(ckpts, str):
            ckpts = [ckpts]
        elif not isinstance(ckpts, (list, tuple)):
            raise ValueError(
                f"Value for key '{model_version}' is of unsupported type {type(ckpts)}. "
                "Expecting a string or a list of strings."
            )

        for ckpt in ckpts:
            if not ckpt.endswith(".ckpt"):
                raise ValueError(f"Checkpoint '{ckpt}' under '{model_version}' does not end with .ckpt")

        clean_data[model_version] = list(ckpts)

    return clean_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit jobs for custom model embedding across all checkpoints.")
    parser.add_argument("--model_dir", type=str, default="/home/dalalt1/compute/Orthrus/exploration_models", help="Directory of the model to embed.")
    parser.add_argument("--config_file", type=str, default="", help="JSON/YAML file mapping model_version -> list[checkpoint]. If provided, discovery flags are ignored.")
    parser.add_argument("--best_only", action='store_true', help="Only use the best checkpoint for each model.")
    parser.add_argument("--best_onward", action='store_true', help="Use the best checkpoint and all checkpoints after it.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recompute the embeddings even if they already exist.")
    parser.add_argument("--dry_run", action='store_true', help="Only print the commands that would be run, without executing them.")
    parser.add_argument("--filter_substr", type=str, help="Evaluate model versions by this substring. If not provided, all model versions will be considered.", default="")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuration handling: decide which checkpoints to embed
    # ------------------------------------------------------------------

    using_config_file = bool(args.config_file)

    # Warn / normalise incompatible flags
    if using_config_file:
        if args.best_only or args.best_onward or args.filter_substr:
            print("Warning: --config_file provided; ignoring --best_only, --best_onward and --filter_substr flags.")
        model_ckpt_map = load_model_ckpt_config(args.config_file)
    else:
        if args.best_only and args.best_onward:
            print("Warning: Both --best_only and --best_onward are set. Only --best_only will be used.")
            args.best_onward = False

    # ------------------------------------------------------------------
    # Main loop over datasets
    # ------------------------------------------------------------------

    for dataset_name in sorted(DATASET_CATALOG.keys()):

        print(dataset_name)

        d = mb.load_dataset(dataset_name)
        emb_dir = Path(d.embedding_dir)

        # ------------------------------------------------------------------
        # Determine (model_version, checkpoint) pairs for this dataset
        # ------------------------------------------------------------------

        if using_config_file:
            # Explicit list branch ------------------------------------------------
            version_ckpt_iter = (
                (mv, ckpt)
                for mv, ckpt_list in model_ckpt_map.items()
                for ckpt in ckpt_list
            )
        else:
            # Discovery branch ----------------------------------------------------
            def discovery_iter():
                for mv in sorted(os.listdir(args.model_dir)):
                    if args.filter_substr and args.filter_substr not in mv:
                        continue
                    for ckpt in get_ckpts_from_dir(
                        os.path.join(args.model_dir, mv),
                        args.best_only,
                        args.best_onward,
                    ):
                        yield mv, ckpt
            version_ckpt_iter = discovery_iter()

        # ------------------------------------------------------------------
        # Iterate over the selected checkpoint pairs
        # ------------------------------------------------------------------

        for model_version, ckpt in version_ckpt_iter:

            print("\tModel version:", model_version)

            model_short_name = (
                model_version + "_" + ckpt.replace(".ckpt", "")
            ).replace("_", "-").replace("-track", "").replace("best-", "")

            target_path = emb_dir / f"{dataset_name}_{model_short_name}.npz"

            if target_path.exists() and not args.force_recompute:
                # Skip if embedding already exists
                continue

            print("\t\tEmbedding checkpoint:", ckpt)

            if args.dry_run:
                print(
                    f"\t\tsbatch ./slurm_script.sh --model_dir {args.model_dir} "
                    f"--model_version {model_version} --checkpoint {ckpt} "
                    f"--dataset_name {dataset_name} --force_recompute {args.force_recompute}"
                )
            else:
                result = subprocess.run(
                    [
                        "sbatch",
                        "./slurm_script.sh",
                        "--model_dir", args.model_dir,
                        "--model_version", model_version,
                        "--checkpoint", ckpt,
                        "--dataset_name", dataset_name,
                        "--force_recompute", str(args.force_recompute),
                    ],
                    capture_output=True,
                    text=True,
                )
                print("\t\t" + result.stdout.strip())