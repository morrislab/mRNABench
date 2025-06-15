import os
import subprocess
from pathlib import Path

import argparse
import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [2541, 413, 411, 412, 2547]
random_seeds = random_seeds + [321, 421, 311, 2516, 2515]

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit jobs for custom model embedding across all checkpoints.")
    parser.add_argument("--model_dir", type=str, default="/home/dalalt1/compute/Orthrus/exploration_models", help="Directory of the model to embed.")
    parser.add_argument("--best_only", action='store_true', help="Only use the best checkpoint for each model.")
    parser.add_argument("--best_onward", action='store_true', help="Use the best checkpoint and all checkpoints after it.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recompute the embeddings even if they already exist.")
    parser.add_argument("--dry_run", action='store_true', help="Only print the commands that would be run, without executing them.")
    parser.add_argument("--canonical_split", action='store_true', help="Use the canonical split for the dataset.")
    parser.add_argument("--filter_substr", type=str, help="Evaluate model versions by this substring. If not provided, all model versions will be considered.", default="")
    args = parser.parse_args()

    if args.best_only and args.best_onward:
        print("Warning: Both --best_only and --best_onward are set. Only --best_only will be used.")
        args.best_onward = False

    lp_format = "{}/result_lp_{}_{}_{}_tcol-{}_split-{}_rs-{}.json"

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        target_cols = dataset_info["target_col"]

        lp_res_folder = mb.load_dataset(dataset_name).dataset_path + "/lp_results"

        print("Dataset name: ", dataset_name)

        for index, target_col in enumerate(target_cols):

            for model_version in sorted(os.listdir(args.model_dir)):

                if args.filter_substr and args.filter_substr not in model_version:
                    continue

                print("\tModel version:", model_version)

                for ckpt in get_ckpts_from_dir(os.path.join(args.model_dir, model_version), args.best_only, args.best_onward):

                    model_short_name = (model_version + "_" + ckpt.replace(".ckpt", "")).replace("_", "-").replace("-track", "").replace("best-", "")

                    if args.canonical_split:
                        valid_split_types = [DATASET_INFO[dataset_name]["split_type"]]
                    else:
                        if "mrl-sample" in dataset_name:
                            valid_split_types = ["default", "hard-kmer", "kmer"]
                        elif "mrl-hl-lbkwk" in dataset_name:
                            valid_split_types = ["default"]
                        elif "lncrna-ess" in dataset_name:
                            valid_split_types = ["default", "hard-kmer", "kmer", "chromosome"]
                        else:
                            valid_split_types = split_types

                    for split_type in valid_split_types:

                        if split_type == 'hard-kmer' and 'lncrna' in dataset_name:
                            continue

                        # print("\t\tSplit type: ", split_type)

                        all_exists = True
                        for seed in random_seeds:
                            out_fn = lp_format.format(
                                lp_res_folder,
                                dataset_name,
                                model_short_name,
                                dataset_info["task"][index],
                                target_col,
                                split_type,
                                seed
                            )

                            if not os.path.exists(out_fn) or args.force_recompute:
                                all_exists = False
                                break


                        if all_exists:
                            # print("\t\tResults already computed. Skipping.")
                            continue


                        if args.dry_run:
                            print(f"\t\tsbatch ./modelversion_slurm.sh --model_short_name {model_short_name} --dataset_name {dataset_name} --task {dataset_info['task'][index]} --target {target_col} --split_type {split_type} --force_recompute {args.force_recompute}")
                        else:
                            result = subprocess.run(
                                [
                                    "sbatch",
                                    "./modelversion_slurm.sh",
                                    "--model_short_name", model_short_name,
                                    "--dataset_name", dataset_name,
                                    "--task", dataset_info["task"][index],
                                    "--target", target_col,
                                    "--split_type", split_type,
                                    "--force_recompute", str(args.force_recompute),
                                ],
                                capture_output=True,
                                text=True
                            )
                            print("\t\t" + result.stdout.strip())