"""Run linear probing for dataset using embeddings from given model."""

import argparse
import os

from mrna_bench.linear_probe.linear_probe import LinearProbe
from mrna_bench.models import MODEL_CATALOG

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target", type=str, default="target")
parser.add_argument("--seq_chunk_overlap", type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":
    model_class = MODEL_CATALOG[args.model_name]

    prober = LinearProbe.init_from_name(
        args.model_name,
        args.model_version,
        args.data,
        target_col=args.target,
        task=args.task,
        seq_chunk_overlap=args.seq_chunk_overlap,
        split_type="default",
        split_ratios=(0.7, 0.15, 0.15),
        eval_all_splits=True
    )

    lp_res_path = prober.dataset.dataset_path + "/lp_results"

    seeds = [2541, 413, 411, 412, 2547]

    for seed in seeds:
        if not os.path.exists(lp_res_path):
            metrics = prober.run_linear_probe(seed, persist=True)
        elif prober.get_output_filename(seed) not in os.listdir(lp_res_path):
            metrics = prober.run_linear_probe(seed, persist=True)
        else:
            print("Results already computed.")
