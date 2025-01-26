"""Run linear probe for model's embeddings on all datasets."""

import argparse

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.datasets.dataset_catalog import DATASET_DEFAULT_TASK
from mrna_bench.embedder.embedder_utils import get_output_filename
from mrna_bench.linear_probe.linear_probe import LinearProbe

parser = argparse.ArgumentParser()
parser.add_argument("--model_short_name", type=str)
parser.add_argument("--target_col", type=str, default="target")
parser.add_argument("--seq_chunk_overlap", type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":

    for dataset_name in DATASET_CATALOG.keys():
        embedding_fn = get_output_filename(
            args.model_short_name,
            dataset_name,
            sequence_chunk_overlap=args.seq_chunk_overlap,
        )

        task = DATASET_DEFAULT_TASK[dataset_name]

        linear_prober = LinearProbe.init_from_embeding(
            embedding_fn + ".npz",
            task=task,
            target_col=args.target_col,
            split_type="homology",
            split_ratios=(0.7, 0.15, 0.15),
            eval_all_splits=True
        )

        seeds = [2541, 413, 411, 412, 2547]
        metrics = linear_prober.linear_probe_multirun(seeds)
        results = linear_prober.compute_multirun_results(metrics, persist=True)
