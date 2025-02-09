"""Compute lp metrics for all embeddings for a dataset for multiple seeds."""

import argparse
import os

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.linear_probe.linear_probe import LinearProbe

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target", type=str, default="target")
args = parser.parse_args()


if __name__ == "__main__":
    dataset = DATASET_CATALOG[args.data]()

    for embedding_fn in os.listdir(dataset.embedding_dir):
        linear_prober = LinearProbe.init_from_embedding(
            embedding_fn,
            task=args.task,
            target_col=args.target,
            split_type="default",
            split_ratios=(0.7, 0.15, 0.15),
            eval_all_splits=True
        )

        seeds = [2541, 413, 411, 412, 2547]
        metrics = linear_prober.load_results(seeds)
        results = linear_prober.compute_multirun_results(metrics, persist=True)
