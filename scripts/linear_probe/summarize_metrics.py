"""Compute lp metrics for all embeddings for a dataset for multiple seeds."""

import argparse
import os

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--split_type", type=str)
parser.add_argument("--target", type=str, default="target")
args = parser.parse_args()


if __name__ == "__main__":
    dataset = DATASET_CATALOG[args.data]()

    for embedding_fn in os.listdir(dataset.embedding_dir):
        dataset_name = embedding_fn.split("_")[0]
        prober = (
            LinearProbeBuilder(dataset_name=dataset_name)
            .fetch_embedding_by_filename(embedding_fn)
            .set_target(args.target)
            .build_splitter(split_type=args.split_type, eval_all_splits=True)
            .build_evaluator(task=args.task)
            .use_persister()
            .build()
        )

        seeds = [2541, 413, 411, 412, 2547]
        metrics = prober.persister.load_multirun_results(seeds)
        results = prober.compute_multirun_results(metrics, persist=True)
