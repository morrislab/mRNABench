import subprocess

import mrna_bench as mb
from pathlib import Path

from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG

if __name__ == "__main__":
    for dataset_name in DATASET_CATALOG.keys():

        if 'ess' not in dataset_name:
            # Skip datasets that are not in the 'ess' category
            continue

        d = mb.load_dataset(dataset_name)
        emb_dir = Path(d.embedding_dir)

        skip_model_keys = [
            # "-utronly",
            # "codonbert",
            "evo2",
            "helix"
        ]
        print(dataset_name)
        for model_name, model_versions in MODEL_VERSION_MAP.items():

            if model_name in ["Evo2", "Helix-mRNA"]:
                # Skip models with specific environment requirements
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

                subprocess.run([
                    "sbatch",
                    "./slurm_script.sh",
                    "--model_class", model_name,
                    "--model_version", model_version,
                    "--dataset_name", dataset_name,
                    "--d_chunk_ind", "0",
                    "--d_num_chunks", "0"
                ])