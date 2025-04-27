"""Launches slurm jobs for all models on specific dataset."""
import argparse
import subprocess

from mrna_bench.models.model_catalog import MODEL_VERSION_MAP


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
args = parser.parse_args()


if __name__ == "__main__":
    for model_name, model_versions in MODEL_VERSION_MAP.items():
        if model_name in ["Evo2", "Helix-mRNA", "AIDO.RNA"]:
            # Skip models with specific environment requirements
            continue

        for model_version in model_versions:
            subprocess.run([
                "sbatch",
                "./slurm_script.sh",
                model_name,
                model_version,
                args.dataset_name,
                "0",
                "0",
            ])
