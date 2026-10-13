"""Run linear probing for dataset using embeddings from given model."""

import argparse
import ast

import numpy as np
import mrna_bench as mb
from mrna_bench.linear_probe.linear_probe_builder import LinearProbeBuilder
from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.embedder import get_embedding_filepath
from mrna_bench.models import MODEL_CATALOG
from mrna_bench.zeroshot import ZeroShotVEP


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


def load_embeddings(dataset, model_short_name):
    """Load one model's persisted dataset embeddings."""
    path = get_embedding_filepath(
        dataset.embedding_dir,
        model_short_name,
        dataset.dataset_name,
    ) + ".npz"
    return np.load(path)["embedding"]


def build_embedding_vep(dataset, embeddings, model_short_name, target):
    """Build embedding VEP without routing through LinearProbeBuilder."""
    persister = LinearProbePersister(
        dataset,
        model_short_name,
        "embedding_vep",
        target,
        "none",
    )
    return ZeroShotVEP.from_embeddings(
        dataset,
        embeddings,
        target_col=target,
        persister=persister,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
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
parser.add_argument(
    "--score_method",
    choices=[
        "causal_likelihood",
        "pseudo_likelihood",
        "masked_marginal",
    ],
)
parser.add_argument(
    "--normalization",
    choices=["mean", "sum"],
    default="sum",
)
parser.add_argument(
    "--attn_implementation",
    choices=["eager", "sdpa", "flash_attention_2"],
)
parser.add_argument("--score_batch_size", type=int, default=16)
parser.add_argument(
    "--kmers",
    type=str,
    default=None,
    help="k-mer lengths for NaiveBaseline, e.g. '[3, 4]'.",
)
args = parser.parse_args()


if __name__ == "__main__":
    model_class = MODEL_CATALOG[args.model_name]
    model_short_name = model_class.get_model_short_name(args.model_version)
    model_options = None
    if args.kmers:
        model_options = mb.load_model(
            args.model_name, args.model_version,
            kmer_list=ast.literal_eval(args.kmers),
        ).model_options

    print("Running linear probe for model:", model_short_name, flush=True)
    print("Dataset:", args.dataset_name, flush=True)
    print("Task:", args.task, flush=True)
    print("Target:", args.target, flush=True)
    print("Split type:", args.split_type, flush=True)
    print("Seeds:", args.seeds, flush=True)

    dataset = mb.load_dataset(args.dataset_name)

    if args.task == "likelihood_vep":
        model = mb.load_model(
            args.model_name,
            args.model_version,
            attn_implementation=args.attn_implementation,
        )
        if args.score_batch_size < 1:
            raise ValueError("score_batch_size must be at least 1.")
        model.sequence_score_batch_size = args.score_batch_size
        persister = LinearProbePersister(
            dataset,
            model_short_name,
            "likelihood_vep",
            args.target,
            "none",
        )
        evaluator = ZeroShotVEP.from_model(
            dataset,
            model,
            score_method=args.score_method,
            target_col=args.target,
            normalization=args.normalization,
            persister=persister,
        )
        if not args.force_recompute and evaluator.result_exists():
            print("Results already computed, skipping.", flush=True)
        else:
            metrics = evaluator.run(persist=True)
            print("Finished:", metrics, flush=True)
        raise SystemExit(0)

    if args.task == "embedding_vep":
        zero_shot = (
            ZeroShotBuilder(dataset=dataset)
            .fetch_embedding_by_model_name(
                model_short_name, model_options=model_options
            )
            .set_target(args.target)
            .use_persister()
            .build()
        )

        if not args.force_recompute and prober.result_exists():
            print("Results already computed, skipping.", flush=True)
        else:
            print("Running zero-shot VEP.", flush=True)
            metrics = prober.run(persist=True)
            print("Finished:", metrics, flush=True)

    else:
        prober = (
            LinearProbeBuilder(dataset_name=args.dataset_name)
            .fetch_embedding_by_model_name(
                model_short_name, model_options=model_options
            )
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
                print("Results already computed for seed:", seed, flush=True)
                continue

            print("Running linear probe for seed:", seed, flush=True)
            metrics = prober.run_linear_probe(seed, persist=True)
            print("Finished.")
