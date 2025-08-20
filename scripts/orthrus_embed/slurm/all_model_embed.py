import subprocess
import os

import mrna_bench as mb
from pathlib import Path

import argparse
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG

def get_step_from_ckpt(ckpt):
    """
    Extract the step number from the checkpoint filename.
    """
    # Example filename: "epoch=0-step=1000.ckpt"
    # Split by '-' and take the last part, then split by '=' and take the last part

    step = ckpt.split('-')[-1].split('=')[-1].split('.')[0]
    return int(step)

def get_ckpts_from_dir(model_dir, choice=None, best_onward=False):
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

    if choice is not None:

        last_ckpt = sorted(ckpt_files, key=lambda x: get_step_from_ckpt(x.stem), reverse=True)[:1]

        if choice == "last":
            # If last_only, return only the last checkpoint
            ckpt_files = last_ckpt
        elif choice == "best":
            # Filter to only include the top checkpoint which either has the prefix best or if not available, the last checkpoint
            top_files = [f for f in ckpt_files if "best" in f.stem]

            if len(top_files) < 1:
                # If no best checkpoint is found, use the last checkpoint
                ckpt_files = last_ckpt

    if not best_onward and not choice:
        # skip the best checkpoint (it's already included in the full list)
        ckpt_files = [f for f in ckpt_files if "best" not in f.stem]

    return sorted([f.name for f in ckpt_files], key=lambda x: get_step_from_ckpt(x))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit jobs for custom model embedding across all checkpoints.")
    parser.add_argument("--model_dir", type=str, default="", help="Directory of the model to embed.")
    parser.add_argument("--best_only", action='store_true', help="Only use the best checkpoint for each model.")
    parser.add_argument("--last_only", action='store_true', help="Only use the last checkpoint for each model.")
    parser.add_argument("--best_onward", action='store_true', help="Use the best checkpoint and all checkpoints after it.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recompute the embeddings even if they already exist.")
    parser.add_argument("--dry_run", action='store_true', help="Only print the commands that would be run, without executing them.")
    parser.add_argument("--filter_substr", type=str, help="Evaluate model versions by this substring. If not provided, all model versions will be considered.", default="")
    args = parser.parse_args()

    if (args.best_only or args.last_only) and args.best_onward:
        print("Warning: Using --best_onward with --best_only or --last_only is not supported. Ignoring --best_onward.")
        args.best_onward = False

    if args.last_only and args.best_only:
        raise ValueError("Cannot use --last_only and --best_only together. Please choose one.")

    choice = None

    if args.best_only:
        choice = "best"
    elif args.last_only:
        choice = "last"

    for dataset_name in sorted(DATASET_CATALOG.keys()):

        print(dataset_name)

        d = mb.load_dataset(dataset_name)
        emb_dir = Path(d.embedding_dir)

        for model_version in sorted(os.listdir(args.model_dir)):

            if args.filter_substr and args.filter_substr not in model_version:
                continue

            print("\tModel version:", model_version)

            for ckpt in get_ckpts_from_dir(os.path.join(args.model_dir, model_version), choice, args.best_onward):

                model_short_name = (model_version + "_" + ckpt.replace(".ckpt", "")).replace("_", "-").replace("-track", "").replace("best-", "")

                if (emb_dir / f"{dataset_name}_{model_short_name}.npz").exists() and not args.force_recompute:
                    # Skip if embedding already exists
                    # print(f"\t\tEmbedding for {ckpt} already exists. Skipping...")
                    continue
                else:
                    print("\t\tEmbedding checkpoint:", ckpt)


                if args.dry_run:
                    print(f"\t\tsbatch ./slurm_script.sh --model_dir {args.model_dir} --model_version {model_version} --checkpoint {ckpt} --dataset_name {dataset_name} --force_recompute {args.force_recompute}")
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
                        text=True
                    )
                    print("\t\t" + result.stdout.strip())

                    if result.stderr:
                        print("\t\tError:", result.stderr.strip())