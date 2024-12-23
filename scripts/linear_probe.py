import argparse

from mrna_bench.linear_probe.linear_probe import LinearProbe
from mrna_bench.models import MODEL_CATALOG

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--seq_chunk_overlap", type=int)
parser.add_argument("--target_task", type=str)
parser.add_argument("--target_col", type=str, default="target")
args = parser.parse_args()


if __name__ == "__main__":
    model_class = MODEL_CATALOG[args.model_name]

    linear_prober = LinearProbe(
        args.model_name,
        args.model_version,
        args.dataset_name,
        seq_chunk_overlap=args.seq_chunk_overlap,
        target_col=args.target_col,
        target_task=args.target_task,
        split_type="homology",
        split_ratios=[0.7, 0.15, 0.15],
        eval_all_splits=True
    )

    seeds = [2541, 413, 411, 412, 2547]
    metrics = linear_prober.linear_probe_multirun(seeds)
    linear_prober.print_multirun_results(metrics)
