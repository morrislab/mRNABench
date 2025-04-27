"""Run linear probe on specified embeddings for a dataset."""

import argparse
import os

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder

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

    prober = (
        LinearProbeBuilder(args.dataset_name)
        .fetch_embedding_by_filename(args.embedding_fn)
        .build_splitter(args.split_type, species="human")
        .build_evaluator(args.task, eval_all_splits=True)
        .set_target(args.target_col)
        .use_persister()
        .build()
    )

    lp_res_path = dataset.dataset_path + "/lp_results"

    seeds = eval(args.seeds)

    for seed in seeds:
        out_fn = prober.persister.get_output_filename(seed)
        if os.path.exists(lp_res_path) and out_fn in os.listdir(lp_res_path):
            print("Results already computed.")
            continue
        metrics = prober.run_linear_probe(seed, persist=True)
