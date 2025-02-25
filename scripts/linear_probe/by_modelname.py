"""Run linear probing for dataset using embeddings from given model."""

import argparse
import os

from mrna_bench.linear_probe.linear_probe import LinearProbe
from mrna_bench.models import MODEL_CATALOG

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target", type=str, default="target")
parser.add_argument("--seq_chunk_overlap", type=int, default=0)
parser.add_argument("--split_type", type=str, default="default")
parser.add_argument("--isoform_resolved", type=str2bool, default=False)
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
        split_type=args.split_type,
        split_ratios=(0.7, 0.15, 0.15),
        eval_all_splits=True,
        ss_map_path="/home/dalalt1/Orthrus_eval/essentiality/lncRNA_homology/output/similarity_results_full.npz",
        threshold=0.75,
        isoform_resolved=args.isoform_resolved,
    )

    lp_res_path = prober.dataset.dataset_path + "/lp_results"

    seeds = [0, 1, 7, 9, 42, 123, 256, 777, 2025, 31415]

    metrics = prober.linear_probe_multirun(seeds, persist=True)
    average_metrics = prober.compute_multirun_results(metrics, persist=True)

    for seed in seeds:
        if not os.path.exists(lp_res_path):
            metrics = prober.run_linear_probe(seed, persist=True)
        elif prober.get_output_filename(seed) not in os.listdir(lp_res_path):
            metrics = prober.run_linear_probe(seed, persist=True)
        else:
            print("Results already computed.")
