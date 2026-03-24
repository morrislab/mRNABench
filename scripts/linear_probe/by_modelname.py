"""Run linear probing for dataset using embeddings from given model."""

import argparse

import numpy as np
import mrna_bench as mb
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder
from mrna_bench.embedder import get_embedding_filepath
from mrna_bench.models import MODEL_CATALOG


# NAIVE BASELINE FEATURES
K = 21824  # number of all possible unique 3-7mers

FEATURE_BLOCKS = {
    "kmer": np.arange(0, K),
    "gc": np.array([K]),
    "cds": np.array([K + 1]),
    "exon": np.array([K + 2]),
}

FEATURE_COMBOS = {
    "kmer": ["kmer"],
    "struct": ["cds", "exon"],
    "gc-struct": ["gc", "cds", "exon"],
    "all": ["kmer", "gc", "cds", "exon"],
}


default_seeds = "[2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--target", type=str, default="target")
parser.add_argument("--split_type", type=str, default="default")
parser.add_argument("--combo", type=str, default=None)
parser.add_argument("--seeds", type=str, default=default_seeds)
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    model_class = MODEL_CATALOG[args.model_name]
    model_short_name = model_class.get_model_short_name(args.model_version)

    print("Running linear probe for model:", model_short_name, flush=True)
    print("Dataset:", args.dataset_name, flush=True)
    print("Task:", args.task, flush=True)
    print("Target:", args.target, flush=True)
    print("Split type:", args.split_type, flush=True)
    print("Seeds:", args.seeds, flush=True)

    dataset = mb.load_dataset(args.dataset_name)


    if args.model_name == "NaiveBaseline":

        embeddings_path = get_embedding_filepath(
            output_dir = dataset.embedding_dir,
            model_short_name = model_short_name,
            dataset_name = args.dataset_name,
        ) + ".npz"

        embeddings = np.load(embeddings_path)["embedding"]

        if model_short_name == "naive-4":

            FEATURE_COMBOS = {
                "gc": ["gc"],
                "kmer": ["kmer"],
                "all": ["kmer", "gc"],
            }

        if args.combo is not None:
            FEATURE_COMBOS = {
                args.combo: FEATURE_COMBOS[args.combo]
            }

        for combo_name, blocks in FEATURE_COMBOS.items():
            print("Running naive baseline for feature combo:", combo_name, flush=True)

            idx = np.concatenate([FEATURE_BLOCKS[feat] for feat in blocks])
            idx = np.sort(idx)

            if combo_name == "all":
                feature_model_name = model_short_name
                embeddings_subset = embeddings.copy()
            else:
                feature_model_name = f"{model_short_name}-{combo_name}"
                embeddings_subset = embeddings[:, idx].copy()

            prober = (
                LinearProbeBuilder(dataset_name=args.dataset_name)
                .fetch_embedding_by_embedding_instance(feature_model_name, embeddings_subset)
                .build_splitter(args.split_type, species=dataset.species, eval_all_splits=True)
                .build_evaluator(args.task)
                .set_target(args.target)
                .use_persister()
                .build()
            )

            seeds = eval(args.seeds)

            for seed in seeds:
                if not args.force_recompute and prober.persister.result_exists(seed):
                    print("Results already computed for seed:", seed, flush=True)
                    continue

                print("Running linear probe for seed:", seed, flush=True)
                metrics = prober.run_linear_probe(seed, persist=True)
                print("Finished.")

    else:
        prober = (
            LinearProbeBuilder(dataset_name=args.dataset_name)
            .fetch_embedding_by_model_name(model_short_name)
            .build_splitter(args.split_type, species=dataset.species, eval_all_splits=True)
            .build_evaluator(args.task)
            .set_target(args.target)
            .use_persister()
            .build()
        )

        seeds = eval(args.seeds)

        for seed in seeds:
            if not args.force_recompute and prober.persister.result_exists(seed):
                print("Results already computed for seed:", seed, flush=True)
                continue

            print("Running linear probe for seed:", seed, flush=True)
            metrics = prober.run_linear_probe(seed, persist=True)
            print("Finished.")
