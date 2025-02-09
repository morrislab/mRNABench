"""Run linear probe on specified embeddings for a dataset."""

import argparse
import os

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.linear_probe.linear_probe import LinearProbe

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_fn", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target_col", type=str, default="target")
parser.add_argument("--split_type", type=str, default="default")
parser.add_argument("--seeds", type=str, default="[2541, 413, 411, 412, 2547]")

args = parser.parse_args()


if __name__ == "__main__":
    dataset = DATASET_CATALOG[args.dataset_name]()

    prober = LinearProbe.init_from_embedding(
        args.embedding_fn,
        task=args.task,
        target_col=args.target_col,
        split_type=args.split_type,
        split_ratios=(0.7, 0.15, 0.15),
        eval_all_splits=True
    )

    lp_res_path = dataset.dataset_path + "/lp_results"

    seeds = eval(args.seeds)

    for seed in seeds:
        if not os.path.exists(lp_res_path):
            metrics = prober.run_linear_probe(seed, persist=True)
        elif prober.get_output_filename(seed) not in os.listdir(lp_res_path):
            metrics = prober.run_linear_probe(seed, persist=True)
        else:
            print("Results already computed.")
