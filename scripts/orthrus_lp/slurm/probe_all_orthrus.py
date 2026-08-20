import os
import subprocess
from pathlib import Path

import argparse
import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]


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

    parser = argparse.ArgumentParser(description="Submit jobs for custom model linear probing.")
    parser.add_argument("--model_dir", type=str, default="", help="Directory of the model to embed.")
    parser.add_argument("--best_only", action='store_true', help="Only use the best checkpoint for each model.")
    parser.add_argument("--last_only", action='store_true', help="Only use the last checkpoint for each model.")
    parser.add_argument("--best_onward", action='store_true', help="Use the best checkpoint and all checkpoints after it.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recompute the embeddings even if they already exist.")
    parser.add_argument("--dry_run", action='store_true', help="Only print the commands that would be run, without executing them.")
    parser.add_argument("--canonical_split", action='store_true', help="Use the canonical split for the dataset.")
    parser.add_argument("--filter_substr", type=str, help="Evaluate model versions by this substring. If not provided, all model versions will be considered.", default="")
    parser.add_argument("--per_seed", action='store_true', help="Submit jobs per random seed.")
    parser.add_argument(
        "--regressor",
        choices=["ols", "ridge"],
        default="ols",
    )
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

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        dataset = mb.load_dataset(dataset_name)

        print("Dataset name: ", dataset_name)

        jobs = [
            (spec.task, spec.target_col)
            for spec in dataset.metadata.task_specs
        ]
        if "embedding_vep" in dataset_info["evaluations"]:
            jobs.extend(
                ("embedding_vep", target)
                for target in dataset.metadata.target_col
            )

        for task, target_col in jobs:

                for model_version in sorted(os.listdir(args.model_dir)):

                    if args.filter_substr and args.filter_substr not in model_version:
                        continue

                    # print("\tModel version:", model_version)

                    for ckpt in get_ckpts_from_dir(os.path.join(args.model_dir, model_version), choice, args.best_onward):

                        model_short_name = (model_version + "_" + ckpt.replace(".ckpt", "")).replace("_", "-").replace("-track", "").replace("best-", "")

                        # ----------------------------
                        # embedding VEP: single run, no splits, no seeds
                        # ----------------------------
                        if task == "embedding_vep":
                            if not args.force_recompute:
                                persister = LinearProbePersister(
                                    dataset,
                                    model_short_name,
                                    "embedding_vep",
                                    target_col,
                                    "none",
                                )
                                if persister.result_exists("embedding_vep"):
                                    continue

                            cmd = [
                                "sbatch",
                                "./modelversion_slurm.sh",
                                "--model_short_name", model_short_name,
                                "--dataset_name", dataset_name,
                                "--task", "embedding_vep",
                                "--target", target_col,
                                "--split_type", "none",
                                "--seeds", '["embedding_vep"]',
                                "--force_recompute", str(args.force_recompute),
                            ]

                            if args.dry_run:
                                print(f"\t\tDry run: {' '.join(cmd)}")
                            else:
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                if result.stdout:
                                    print('\t' + result.stdout.strip())
                                if result.stderr:
                                    print('\t' + result.stderr.strip())
                            continue

                        # ----------------------------
                        # standard LP: splits + seeds
                        # ----------------------------
                        if args.canonical_split:
                            valid_split_types = [DATASET_INFO[dataset_name]["default_split_type"]]
                        elif "mrl-sample" in dataset_name:
                            valid_split_types = ["default", "hard-kmer", "kmer"]
                        elif "mrl-hl-lbkwk" in dataset_name:
                            valid_split_types = ["default"]
                        else:
                            valid_split_types = split_types

                        for split_type in valid_split_types:

                            # ----------------------------
                            # decide seeds
                            # ----------------------------
                            seeds_to_run = random_seeds if args.per_seed else [random_seeds]

                            for seed_block in seeds_to_run:

                                seeds = [seed_block] if args.per_seed else seed_block

                                if not args.force_recompute:
                                    persister = LinearProbePersister(
                                        dataset,
                                        model_short_name,
                                        task,
                                        target_col,
                                        split_type,
                                        regressor=args.regressor,
                                    )
                                    if all(persister.result_exists(seed) for seed in seeds):
                                        continue

                                seed_arg = f"[{seeds[0]}]" if args.per_seed else str(seeds)

                                cmd = [
                                    "sbatch",
                                    "./modelversion_slurm.sh",
                                    "--model_short_name", model_short_name,
                                    "--dataset_name", dataset_name,
                                    "--task", task,
                                    "--regressor", args.regressor,
                                    "--target", target_col,
                                    "--split_type", split_type,
                                    "--seeds", seed_arg,
                                    "--force_recompute", str(args.force_recompute),
                                ]

                                if args.dry_run:
                                    print(f"\t\tDry run: {' '.join(cmd)}")
                                else:
                                    result = subprocess.run(
                                        cmd,
                                        capture_output=True,
                                        text=True
                                    )
                                    if result.stdout:
                                        print("\t\t" + result.stdout.strip())
                                    if result.stderr:
                                        print("\t\t" + result.stderr.strip())
