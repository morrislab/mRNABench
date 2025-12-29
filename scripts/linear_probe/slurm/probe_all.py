import os
import subprocess
import argparse

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs for model evaluation.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of results.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing.")
    parser.add_argument("--canonical_split", action='store_true', help="Use canonical split for each dataset.")
    args = parser.parse_args()

    lp_format = "{}/result_lp_{}_{}_{}_tcol-{}_split-{}_rs-{}.json"

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        if 'vep' in dataset_name or 'utr' in dataset_name:
            continue

        target_cols = dataset_info["target_col"]
        lp_res_folder = mb.load_dataset(dataset_name).dataset_path + "/lp_results"

        print("Dataset name: ", dataset_name)

        for index, target_col in enumerate(target_cols):
            for model_name, model_versions in MODEL_VERSION_MAP.items():

                for model_version in model_versions:
                    if model_version in ["evo2_40b_base", "evo2_40b"]:
                        continue

                    model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                    if args.canonical_split:
                        valid_split_types = [DATASET_INFO[dataset_name]["default_split_type"]]
                    elif "mrl-sample" in dataset_name:
                        valid_split_types = ["default", "hard-kmer", "kmer"]
                    elif "mrl-hl-lbkwk" in dataset_name:
                        valid_split_types = ["default"]
                    elif "utr-variants" in dataset_name:
                        valid_split_types = ["default"]
                    else:
                        valid_split_types = split_types

                    for split_type in valid_split_types:

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
                            # print("Results already computed. Skipping.")
                            continue

                        cmd = [
                            "sbatch",
                            "./modelname_slurm.sh",
                            "--model_name", model_name,
                            "--model_version", model_version,
                            "--dataset_name", dataset_name,
                            "--task", dataset_info["task"][index],
                            "--target", target_col,
                            "--split_type", split_type,
                            "--force_recompute", str(args.force_recompute),
                        ]

                        if args.dry_run:
                            print("\t\tDry run:", " ".join(cmd))
                        else:
                            result = subprocess.run(cmd, capture_output=True, text=True)

                            print('\t' + result.stdout.strip())
