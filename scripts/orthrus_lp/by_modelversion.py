"""Run linear probing for dataset using embeddings from given model."""

import argparse
import ast

import mrna_bench as mb
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder
from mrna_bench.zeroshot import ZeroShotBuilder

default_seeds = "[2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]"

parser = argparse.ArgumentParser()
parser.add_argument("--model_short_name", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--task", type=str)
parser.add_argument(
    "--regressor",
    choices=["ols", "ridge"],
    default="ridge",
)
parser.add_argument("--target", type=str, default="target")
parser.add_argument("--split_type", type=str, default="default")
parser.add_argument("--seeds", type=str, default=default_seeds)
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":

    print("Running linear probe for model:", args.model_short_name)
    print("Dataset:", args.dataset_name)
    print("Task:", args.task)
    print("Target:", args.target)
    print("Split type:", args.split_type)
    print("Seeds:", args.seeds)

    dataset = mb.load_dataset(args.dataset_name)

    if args.task == "embedding_vep":
        zero_shot = (
            ZeroShotBuilder(dataset=dataset)
            .fetch_embedding_by_model_name(args.model_short_name)
            .set_target(args.target)
            .use_persister()
            .build()
        )

        if not args.force_recompute and zero_shot.result_exists():
            print("Results already computed, skipping.")
        else:
            print("Running embedding VEP.")
            metrics = zero_shot.run_zero_shot(persist=True)
            print("Finished:", metrics)

    else:
        prober = (
            LinearProbeBuilder(dataset_name=args.dataset_name)
            .fetch_embedding_by_model_name(args.model_short_name)
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
            print("Finished")
