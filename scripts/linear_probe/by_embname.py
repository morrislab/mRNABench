"""Run linear probe on specified embeddings for a dataset."""

import argparse
import ast

import numpy as np

from mrna_bench.datasets import DATASET_CATALOG
from mrna_bench.embedder import get_embedding_filepath
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder
from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.zeroshot import ZeroShotVEP

default_seeds = "[2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]"

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_fn", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--task", type=str)
parser.add_argument(
    "--regressor",
    choices=["ols", "ridge"],
    default="ols",
)
parser.add_argument("--target", type=str, default="target")
parser.add_argument("--split_type", type=str, default="default")
parser.add_argument("--seeds", type=str, default=default_seeds)
parser.add_argument("--force_recompute", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    dataset = DATASET_CATALOG[args.dataset_name]()

    if args.task == "embedding_vep":
        name_parts = args.embedding_fn.removesuffix(".npz").split("_")
        if len(name_parts) < 2:
            raise ValueError(
                "Invalid embedding filename: {}".format(args.embedding_fn)
            )
        model_short_name = name_parts[1]
        path = get_embedding_filepath(
            dataset.embedding_dir,
            model_short_name,
            dataset.dataset_name,
        ) + ".npz"
        persister = LinearProbePersister(
            dataset,
            model_short_name,
            "embedding_vep",
            args.target,
            "none",
        )
        evaluator = ZeroShotVEP.from_embeddings(
            dataset,
            np.load(path)["embedding"],
            target_col=args.target,
            persister=persister,
        )
        if args.force_recompute or not evaluator.result_exists():
            evaluator.run(persist=True)
        raise SystemExit(0)

    prober = (
        LinearProbeBuilder(dataset_name=args.dataset_name)
        .fetch_embedding_by_filename(args.embedding_fn)
        .build_splitter(args.split_type, species=dataset.metadata.species, eval_all_splits=True)
        .build_evaluator(args.task)
        .set_regressor(args.regressor)
        .set_target(args.target)
        .use_persister()
        .build()
    )

    seeds = ast.literal_eval(args.seeds)

    for seed in seeds:
        if not args.force_recompute and prober.persister.result_exists(seed):
            print("Results already computed for seed:", seed)
            continue

        print("Running linear probe for seed:", seed)
        metrics = prober.run_linear_probe(seed, persist=True)
