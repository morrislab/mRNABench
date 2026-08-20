"""Submit SLURM jobs for fine-tuning across model/dataset combos."""

import os
import subprocess
import argparse

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [0]
learning_rates = [1e-5, 1e-4, 1e-3]
lora_ranks = [4, 8, 16]
lora_alphas = [8, 16, 32]
accumulation_steps = 8

# Models that are too large or not suitable for LoRA fine-tuning
SKIP_VERSIONS = {"evo2_40b_base", "evo2_40b"}
SKIP_MODELS = {"NaiveBaseline", "NaiveBaselineSixTrack"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for fine-tuning."
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Force recomputation of existing results.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--canonical_split", action="store_true",
        help="Use only the default split for each dataset.",
    )
    parser.add_argument(
        "--eval_test", action="store_true",
        help="Also evaluate on test set.",
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="none",
        choices=["none", "linear", "cosine"],
        help="Learning rate schedule type.",
    )
    parser.add_argument(
        "--dataset_name", nargs="+", default=None,
        help="If specified, only fine-tune on these datasets.",
    )
    parser.add_argument(
        "--model_name", nargs="+", default=None,
        help="If specified, only fine-tune these models.",
    )
    parser.add_argument(
        "--model_version", nargs="+", default=None,
        help="If specified, only fine-tune these model versions.",
    )
    args = parser.parse_args()

    from mrna_bench.fine_tune import FineTunePersister

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        if (
            args.dataset_name is not None
            and dataset_name not in args.dataset_name
        ):
            continue

        if "embedding_vep" in dataset_info["evaluations"]:
            continue

        dataset = mb.load_dataset(dataset_name)

        print("Dataset: {}".format(dataset_name))

        for task_spec in dataset.metadata.task_specs:
            target_col = task_spec.target_col
            task = task_spec.task

            for model_name, model_versions in MODEL_VERSION_MAP.items():
                if model_name in SKIP_MODELS:
                    continue

                if (
                    args.model_name is not None
                    and model_name not in args.model_name
                ):
                    continue

                for model_version in model_versions:
                    if model_version in SKIP_VERSIONS:
                        continue

                    if (
                        args.model_version is not None
                        and model_version not in args.model_version
                    ):
                        continue

                    model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                    if args.canonical_split:
                        valid_split_types = [dataset_info["default_split_type"]]
                    elif "mrl-sample" in dataset_name:
                        valid_split_types = ["default", "hard-kmer", "kmer"]
                    elif "mrl-hl-lbkwk" in dataset_name:
                        valid_split_types = ["default"]
                    else:
                        valid_split_types = split_types

                    for split_type in valid_split_types:
                        for lr in learning_rates:
                            for lora_rank, lora_alpha in zip(lora_ranks, lora_alphas):
                                persister = FineTunePersister(
                                    dataset=dataset,
                                    model_short_name=model_short_name,
                                    task=task,
                                    target_col=target_col,
                                    split_type=split_type,
                                    learning_rate=lr,
                                    lora_rank=lora_rank,
                                    lora_alpha=lora_alpha,
                                )

                                all_exists = True
                                for seed in random_seeds:
                                    path = persister._get_path(seed, ".json")
                                    if not path.exists() or args.force_recompute:
                                        all_exists = False
                                        break

                                if all_exists:
                                    continue

                                cmd = [
                                    "sbatch",
                                    "./finetune_slurm.sh",
                                    "--model_name", model_name,
                                    "--model_version", model_version,
                                    "--dataset_name", dataset_name,
                                    "--task", task,
                                    "--target", target_col,
                                    "--split_type", split_type,
                                    "--seeds", str(random_seeds),
                                    "--learning_rates", str([lr]),
                                    "--lr_schedule", args.lr_schedule,
                                    "--lora_ranks", str([lora_rank]),
                                    "--lora_alphas", str([lora_alpha]),
                                    "--accumulation_steps", str(accumulation_steps),
                                    "--force_recompute", str(args.force_recompute),
                                    "--eval_test", str(args.eval_test),
                                ]

                                if args.dry_run:
                                    print("\t\tDry run: {}".format(" ".join(cmd)))
                                else:
                                    result = subprocess.run(
                                        cmd, capture_output=True, text=True
                                    )
                                    print("\t{}".format(result.stdout.strip()))
