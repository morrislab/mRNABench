import os
import subprocess
import argparse

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

# NAIVE BASELINE FEATURES
K = 21824  # number of all possible unique 3-7mers

MODEL_FEATURE_COMBOS = {
    "naive-4": ["gc", "kmer", "all"],
    "naive-6": ["kmer", "struct", "gc-struct", "all"],
}

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs for model evaluation.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of results.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing.")
    parser.add_argument("--canonical_split", action='store_true', help="Use canonical split for each dataset.")
    parser.add_argument("--per_seed", action='store_true', help="Submit jobs per random seed.")
    args = parser.parse_args()

    lp_format = "{}/result_lp_{}_{}_{}_tcol-{}_split-{}_rs-{}.json"

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        target_cols = dataset_info["target_col"]
        lp_res_folder = mb.load_dataset(dataset_name).dataset_path + "/lp_results"

        print("Dataset name: ", dataset_name)

        skip_model_keys = [
            "evo2-40b",
            "replicate" # skip the borzoi and flashzoi replicate models
        ]

        for index, target_col in enumerate(target_cols):
            for model_name, model_versions in MODEL_VERSION_MAP.items():
                for model_version in model_versions:

                    model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                    if sum([k in model_short_name for k in skip_model_keys]) > 0:
                        continue

                    if args.canonical_split:
                        valid_split_types = [DATASET_INFO[dataset_name]["split_type"]]
                    elif "mrl-sample" in dataset_name:
                        valid_split_types = ["default", "hard-kmer", "kmer"]
                    elif "mrl-hl-lbkwk" in dataset_name:
                        valid_split_types = ["default"]
                    else:
                        valid_split_types = split_types

                    for split_type in valid_split_types:

                        # ----------------------------
                        # decide feature combos
                        # ----------------------------
                        if model_name == "NaiveBaseline" and model_short_name in MODEL_FEATURE_COMBOS:
                            combos = MODEL_FEATURE_COMBOS[model_short_name]
                        else:
                            combos = [""]

                        # ----------------------------
                        # decide seeds
                        # ----------------------------
                        seeds_to_run = random_seeds if args.per_seed else [random_seeds]

                        for combo in combos:
                            for seed_block in seeds_to_run:

                                seeds = [seed_block] if args.per_seed else seed_block

                                # ----------------------------
                                # existence check
                                # ----------------------------
                                if not args.force_recompute:
                                    all_exist = True
                                    for seed in seeds:

                                        if combo and combo != "all":
                                            name = f"{model_short_name}-{combo}"
                                        else:
                                            name = model_short_name

                                        out_fn = lp_format.format(
                                            lp_res_folder,
                                            dataset_name,
                                            name,
                                            dataset_info["task"][index],
                                            target_col,
                                            split_type,
                                            seed
                                        )

                                        if not os.path.exists(out_fn):
                                            all_exist = False
                                            break

                                    if all_exist:
                                        continue

                                seed_arg = f"[{seeds[0]}]" if args.per_seed else str(seeds)

                                cmd = [
                                    "sbatch",
                                    "./modelname_slurm.sh",
                                    "--model_name", model_name,
                                    "--model_version", model_version,
                                    "--dataset_name", dataset_name,
                                    "--task", dataset_info["task"][index],
                                    "--target", target_col,
                                    "--split_type", split_type,
                                    "--combo", combo,
                                    "--seeds", seed_arg,
                                    "--force_recompute", str(args.force_recompute),
                                ]

                                if args.dry_run:
                                    print("\t\tDry run:", " ".join(cmd))
                                else:
                                    result = subprocess.run(cmd, capture_output=True, text=True)

                                    if result.stdout:
                                        print('\t' + result.stdout.strip())
                                    if result.stderr:
                                        print('\t' + result.stderr.strip())
