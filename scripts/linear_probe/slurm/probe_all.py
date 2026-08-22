import subprocess
import argparse

import mrna_bench as mb
from mrna_bench.datasets.dataset_catalog import DATASET_INFO
from mrna_bench.linear_probe.persister import LinearProbePersister
from mrna_bench.models import ModelBehavior
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.data_splitter.split_catalog import SPLIT_CATALOG
from mrna_bench.zeroshot import ZeroShotVEP

# NAIVE BASELINE FEATURES
K = 21824  # number of all possible unique 3-7mers

MODEL_FEATURE_COMBOS = {
    "naive-4": ["gc", "kmer", "all"],
    "naive-6": ["kmer", "struct", "gc-struct", "all"],
}

split_types = list(SPLIT_CATALOG.keys())

random_seeds = [2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs for model evaluation.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of results.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing.")
    parser.add_argument("--canonical_split", action='store_true', help="Use canonical split for each dataset.")
    parser.add_argument("--per_seed", action='store_true', help="Submit jobs per random seed.")
    parser.add_argument(
        "--likelihood_vep",
        action="store_true",
        help="Submit GPU likelihood-VEP jobs for VEP datasets.",
    )
    parser.add_argument(
        "--likelihood_attn",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Override each likelihood model's default attention backend.",
    )
    parser.add_argument("--score_batch_size", type=int, default=16)
    parser.add_argument(
        "--regressor",
        choices=["ols", "ridge"],
        default="ols",
    )
    parser.add_argument("--model_version", type=str, default=None, help="Specify a model version to run. If not provided, all versions will be run.")
    args = parser.parse_args()

    for _, dataset_info in DATASET_INFO.items():
        dataset_name = dataset_info["dataset"]

        print("Dataset name: ", dataset_name)

        dataset = mb.load_dataset(dataset_name)

        skip_model_keys = [
            "evo2-40b-base",
            "evo2-40b",
            "replicate", # skip the individual borzoi and flashzoi replicate models
            "aido-dna",
            "aido-rna-mars",
            "carbon",
            "utrlm-base",
            "utrlm-ss",
            "rnaernie2",
            "dnabert-3mer", "dnabert-4mer", "dnabert-5mer", "dnabert-6mer",
            "ernierna-mrl",
        ]

        standard_jobs = [
            (spec.task, spec.target_col)
            for spec in dataset.metadata.task_specs
        ]
        vep_spec = dataset.metadata.vep_task_spec
        vep_jobs = (
            [(vep_spec.task, vep_spec.target_col)]
            if vep_spec is not None
            else []
        )
        jobs = list(dict.fromkeys(standard_jobs + vep_jobs))
        if (
            "embedding_vep" in dataset_info["evaluations"]
            and vep_spec is not None
            and vep_spec.task != "regression"
        ):
            jobs.append(("embedding_vep", vep_spec.target_col))
        if (
            args.likelihood_vep
            and "likelihood_vep" in dataset_info["evaluations"]
            and vep_spec is not None
        ):
            jobs.append(("likelihood_vep", vep_spec.target_col))

        for target_col in dict.fromkeys(
            target for _, target in jobs
        ):
            tasks = [
                task
                for task, job_target in jobs
                if job_target == target_col
            ]
            for task in tasks:
                for model_name, model_versions in MODEL_VERSION_MAP.items():
                    for model_version in model_versions:
                        if args.model_version is not None and model_version != args.model_version:
                            continue

                        model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                        if sum([k in model_short_name for k in skip_model_keys]) > 0:
                            continue

                        if task == "likelihood_vep":
                            model_class = MODEL_CATALOG[model_name]
                            if (
                                args.likelihood_attn is not None
                                and (
                                    model_class.valid_attn_implementations
                                    is None
                                    or args.likelihood_attn not in
                                    model_class.valid_attn_implementations
                                )
                            ):
                                continue
                            scope = getattr(
                                model_class, "sequence_score_scope", "full"
                            )
                            region = dataset.metadata.variant_region
                            if scope != "full" and scope != region:
                                continue
                            behaviors = model_class.behaviors_for_version(
                                model_version
                            )
                            methods = []
                            if ModelBehavior.CAUSAL_LIKELIHOOD in behaviors:
                                methods.append("causal_likelihood")
                            if ModelBehavior.PSEUDO_LIKELIHOOD in behaviors:
                                methods.extend([
                                    "pseudo_likelihood",
                                    "masked_marginal",
                                ])
                            for method in methods:
                                effective_attn = (
                                    args.likelihood_attn
                                    if args.likelihood_attn is not None
                                    else (
                                        model_class
                                        .default_attn_implementation
                                    )
                                )
                                result_key = ZeroShotVEP.likelihood_result_key(
                                    method,
                                    "sum",
                                    effective_attn or "none",
                                    ZeroShotVEP.default_likelihood_direction(
                                        vep_spec.task
                                    ),
                                )
                                persister = LinearProbePersister(
                                    dataset,
                                    model_short_name,
                                    "likelihood_vep",
                                    target_col,
                                    "none",
                                )
                                if (
                                    not args.force_recompute
                                    and persister.result_exists(result_key)
                                ):
                                    continue
                                cmd = [
                                    "sbatch",
                                    "./likelihood_vep_slurm.sh",
                                    "--model_name", model_name,
                                    "--model_version", model_version,
                                    "--dataset_name", dataset_name,
                                    "--target", target_col,
                                    "--score_method", method,
                                    "--normalization", "sum",
                                    "--score_batch_size",
                                    str(args.score_batch_size),
                                    "--force_recompute",
                                    str(args.force_recompute),
                                ]
                                if args.likelihood_attn is not None:
                                    cmd.extend([
                                        "--attn_implementation",
                                        args.likelihood_attn,
                                    ])
                                if args.dry_run:
                                    print(
                                        "\t\tDry run:", " ".join(cmd)
                                    )
                                else:
                                    result = subprocess.run(
                                        cmd,
                                        capture_output=True,
                                        text=True,
                                    )
                                    if result.stdout:
                                        print(
                                            "\t" + result.stdout.strip()
                                        )
                                    if result.stderr:
                                        print(
                                            "\t" + result.stderr.strip()
                                        )
                            continue

                        # ----------------------------
                        # embedding VEP: single run, no splits, no seeds
                        # ----------------------------
                        if task == "embedding_vep":
                            model_class = MODEL_CATALOG[model_name]
                            scope = getattr(
                                model_class, "sequence_score_scope", "full"
                            )
                            region = dataset.metadata.variant_region
                            if scope != "full" and scope != region:
                                continue
                            if "NaiveBaseline" in model_name and model_short_name in MODEL_FEATURE_COMBOS:
                                combos = MODEL_FEATURE_COMBOS[model_short_name]
                            else:
                                combos = [""]

                            for combo in combos:
                                if not args.force_recompute:
                                    if combo and combo != "all":
                                        check_name = f"{model_short_name}-{combo}"
                                    else:
                                        check_name = model_short_name

                                    persister = LinearProbePersister(
                                        dataset,
                                        check_name,
                                        "embedding_vep",
                                        target_col,
                                        "none",
                                    )
                                    if persister.result_exists(
                                        "embedding_vep"
                                    ):
                                        continue

                                cmd = [
                                    "sbatch",
                                    "./modelname_slurm.sh",
                                    "--model_name", model_name,
                                    "--model_version", model_version,
                                    "--dataset_name", dataset_name,
                                    "--task", "embedding_vep",
                                    "--target", target_col,
                                    "--split_type", "none",
                                    "--combo", combo,
                                    "--seeds", '["embedding_vep"]',
                                    "--force_recompute", str(args.force_recompute),
                                ]

                                if args.dry_run:
                                    print("\t\tDry run:", " ".join(cmd))
                                else:
                                    result = subprocess.run(cmd, capture_output=True, text=True)
                                    if result.stdout:
                                        print('\t' + result.stdout.strip())
                                    if result.stderr:
                                        print('\t' + result.stderr.strip())
                            continue

                        # ----------------------------
                        # standard LP: splits + seeds
                        # ----------------------------
                        if args.canonical_split:
                            valid_split_types = [DATASET_INFO[dataset_name]["default_split_type"]]
                        elif "mrl-sample" in dataset_name or "apa-isoform" in dataset_name or "ires-classification" in dataset_name:
                            valid_split_types = ["default", "hard-kmer", "kmer"]
                        elif "mrl-hl-lbkwk" in dataset_name:
                            valid_split_types = ["default"]
                        else:
                            valid_split_types = split_types

                        for split_type in valid_split_types:

                            # ----------------------------
                            # decide feature combos
                            # ----------------------------
                            if "NaiveBaseline" in model_name and model_short_name in MODEL_FEATURE_COMBOS:
                                combos = MODEL_FEATURE_COMBOS[model_short_name]
                            else:
                                combos = [""]

                            # ----------------------------
                            # decide seeds
                            # ----------------------------
                            seeds_to_run = random_seeds if args.per_seed else [random_seeds]

                            for combo in combos:
                                for seed_block in seeds_to_run:

                                    seeds = [seed_block] if args.per_seed else seed_block

                                    # ----------------------------
                                    # existence check
                                    # ----------------------------
                                    if not args.force_recompute:
                                        if combo and combo != "all":
                                            check_name = f"{model_short_name}-{combo}"
                                        else:
                                            check_name = model_short_name

                                        persister = LinearProbePersister(
                                            dataset,
                                            check_name,
                                            task,
                                            target_col,
                                            split_type,
                                            regressor=args.regressor,
                                        )
                                        if all(persister.result_exists(seed) for seed in seeds):
                                            continue

                                    seed_arg = f"[{seeds[0]}]" if args.per_seed else str(seeds)

                                    cmd = [
                                        "sbatch",
                                        "./modelname_slurm.sh",
                                        "--model_name", model_name,
                                        "--model_version", model_version,
                                        "--dataset_name", dataset_name,
                                        "--task", task,
                                        "--regressor", args.regressor,
                                        "--target", target_col,
                                        "--split_type", split_type,
                                        "--combo", combo,
                                        "--seeds", seed_arg,
                                        "--force_recompute", str(args.force_recompute),
                                    ]

                                    if args.dry_run:
                                        print("\t\tDry run:", " ".join(cmd))
                                    else:
                                        result = subprocess.run(cmd, capture_output=True, text=True)

                                        if result.stdout:
                                            print('\t' + result.stdout.strip())
                                        if result.stderr:
                                            print('\t' + result.stderr.strip())
