import os
import subprocess

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [2541, 413, 411, 412, 2547]
random_seeds = random_seeds + [321, 421, 311, 2516, 2515]

if __name__ == "__main__":
    lp_format = "{}/result_lp_{}_{}_{}_tcol-{}_split-{}_rs-{}.json"

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        if 'ess' not in dataset_name:
            # Skip datasets that are not in the 'ess' category
            continue

        if dataset_name == "lncrna-ess-shared":
            print("Skipping shared dataset")
            continue

        target_cols = dataset_info["target_col"]

        lp_res_folder = mb.load_dataset(dataset_name).dataset_path + "/lp_results"

        print("Dataset name: ", dataset_name)

        for index, target_col in enumerate(target_cols):
            for model_name, model_versions in MODEL_VERSION_MAP.items():
                print("Model name: ", model_name)

                for model_version in model_versions:
                    if model_version in ["evo2_40b_base", "evo2_40b", "helix-mrna"]:
                        continue

                    model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                    if "mrl-sample" in dataset_name:
                        valid_split_types = ["default", "hard-kmer", "kmer"]
                    elif "mrl-hl-lbkwk" in dataset_name:
                        valid_split_types = ["default"]
                    elif "lncrna-ess" in dataset_name:
                        valid_split_types = ["default", "hard-kmer", "kmer", "chromosome"] # drop homology
                    else:
                        valid_split_types = split_types

                    for split_type in valid_split_types:

                        if split_type == 'hard-kmer': # skip hard-kmer for lncrna
                            continue

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

                            if not os.path.exists(out_fn):
                                all_exists = False
                                break

                        if all_exists:
                            print("Results already computed. Skipping.")
                            continue

                        subprocess.run([
                            "sbatch",
                            "./modelname_slurm.sh",
                            model_name,
                            model_version,
                            dataset_name,
                            dataset_info["task"][index],
                            target_col,
                            split_type
                        ])