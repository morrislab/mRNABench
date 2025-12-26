import subprocess

import argparse
import mrna_bench as mb
from pathlib import Path

from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit jobs for model embedding.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of results.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing.")
    args = parser.parse_args()

    for dataset_name in DATASET_CATALOG.keys():

        print(dataset_name)
        d = mb.load_dataset(dataset_name)
        emb_dir = Path(d.embedding_dir)

        skip_model_keys = [
            # "-utronly",
            # "codonbert",
            "evo2",
            # "helix"
        ]

        for model_name, model_versions in MODEL_VERSION_MAP.items():

            if model_name in ["Evo2"]:#, "Helix-mRNA"]:
                # Skip models with specific environment/GPU requirements
                continue

            for model_version in model_versions:
                model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                if sum([k in model_short_name for k in skip_model_keys]) > 0:
                    continue

                if (emb_dir / f"{dataset_name}_{model_short_name}.npz").exists():
                    # Skip if embedding already exists
                    # print(f"Embedding for {dataset_name} with {model_name} {model_version} already exists.")
                    continue

                print(f"Submitting job for {dataset_name} with {model_name} {model_version}")

                cmd = [
                    "sbatch",
                    "./slurm_script.sh",
                    "--model_name", model_name,
                    "--model_version", model_version,
                    "--dataset_name", dataset_name,
                    "--d_chunk_ind", "0",
                    "--d_num_chunks", "0",
                    "--force_recompute", str(args.force_recompute),
                ]

                if args.dry_run:
                    print("\tDry run:", " ".join(cmd))
                else:

                    result = subprocess.run(cmd, capture_output=True, text=True)

                    print('\t' + result.stdout.strip())

                    if result.stderr:
                        print('\tError:', result.stderr.strip())
