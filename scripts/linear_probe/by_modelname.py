"""Run linear probing for dataset using embeddings from given model."""

import argparse
import os

from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder
from mrna_bench.models import MODEL_CATALOG

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target", type=str, default="target")
parser.add_argument("--split_type", type=str, default="default")
parser.add_argument("--seq_chunk_overlap", type=int, default=0)
parser.add_argument("--seeds", type=str, default="[2541, 413, 411, 412, 2547]")
args = parser.parse_args()


if __name__ == "__main__":
    model_class = MODEL_CATALOG[args.model_name]
    model_short_name = model_class.get_short_name(args.model_version)

    prober = (
        LinearProbeBuilder(args.data)
        .fetch_embedding_by_model_name(
            model_short_name,
            args.seq_chunk_overlap)
        .build_splitter(
            args.split_type,
            eval_all_splits=True,
            species=["human"])
        .build_evaluator(args.task)
        .set_target(args.target)
        .use_persister()
        .build()
    )

    lp_res_path = prober.dataset.dataset_path + "/lp_results"

    seeds = eval(args.seeds)

    for seed in seeds:
        out_fn = prober.persister.get_output_filename(seed)
        if os.path.exists(lp_res_path) and out_fn in os.listdir(lp_res_path):
            print("Results already computed.")
            continue
        metrics = prober.run_linear_probe(seed, persist=True)
