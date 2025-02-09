import argparse
import os
import subprocess

from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target", type=str, default="target")
args = parser.parse_args()

if __name__ == "__main__":
    dataset = DATASET_CATALOG[args.dataset]()

    for embedding_fn in os.listdir(dataset.embedding_dir):
        subprocess.run([
            "sbatch",
            "./embname_slurm.sh",
            embedding_fn,
            args.dataset,
            args.task,
            args.target
        ])
