"""Run linear probe on all generated embeddings for a dataset."""

import argparse
import os

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.linear_probe.linear_probe import LinearProbe

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target_col", type=str, default="target")
args = parser.parse_args()


if __name__ == "__main__":
    dataset = DATASET_CATALOG[args.dataset_name]()

    for embedding_fn in os.listdir(dataset.embedding_dir):
        linear_prober = LinearProbe.init_from_embeding(
            embedding_fn,
            task=args.task,
            target_col=args.target_col,
            split_type="homology",
            split_ratios=(0.7, 0.15, 0.15),
            eval_all_splits=True
        )

        seeds = [2541, 413, 411, 412, 2547]
        metrics = linear_prober.linear_probe_multirun(seeds)
        results = linear_prober.compute_multirun_results(metrics, persist=True)
