import subprocess

import argparse
import mrna_bench as mb
from pathlib import Path

from mrna_bench.embedder import get_embedding_filepath
from mrna_bench.models.model_catalog import MODEL_VERSION_MAP, MODEL_CATALOG
from mrna_bench.datasets.dataset_catalog import DATASET_CATALOG

DATASET_CHUNKINGS = {
    "mrl-sample-egfp": 25,
    "mrl-sample-mcherry": 25,
    "mrl-sample-designed": 25,
    "mrl-sample-varying": 25,
    "eclip-binding-hepg2": 3,
    "eclip-binding-k562": 3,
    "mirna-target": 4,
    "go-cc": 4,
    "go-mf": 4,
    "rnahl-human": 4,
    "rna-lifecycle-ietswaart": 4,
    "translation-efficiency-human": 4,
    "translation-efficiency-mouse": 4,
    "apa-isoform": 25,
    "ires-classification": 10,
}

def make_and_run_command(
    model_name,
    model_version,
    dataset_name,
    d_chunk_ind,
    d_num_chunks,
    batch_size,
    dry_run=False,
    force_recompute=False,
):

    cmd = [
        "sbatch",
        "./{}".format("evo_slurm_script.sh" if "evo2" in model_name.lower() else "slurm_script.sh"),
        "--model_name", model_name,
        "--model_version", model_version,
        "--dataset_name", dataset_name,
        "--d_chunk_ind", str(d_chunk_ind),
        "--d_num_chunks", str(d_num_chunks),
        "--batch_size", str(batch_size),
        "--force_recompute", str(force_recompute),
    ]

    if dry_run:
        print("\tDry run:", " ".join(cmd))
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)

        print('\t' + result.stdout.strip())

        if result.stderr:
            print('\tError:', result.stderr.strip())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit jobs for model embedding.")
    parser.add_argument("--force_recompute", action='store_true', help="Force recomputation of results.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_version", type=str, default=None, help="Specify a model version to run. If not provided, all versions will be run.")
    args = parser.parse_args()

    for dataset_name in DATASET_CATALOG.keys():

        print(dataset_name)
        d = mb.load_dataset(dataset_name)
        emb_dir = Path(d.embedding_dir)

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

        for model_name, model_versions in MODEL_VERSION_MAP.items():

            for model_version in model_versions:
                if args.model_version is not None and model_version != args.model_version:
                    continue

                model_short_name = MODEL_CATALOG[model_name].get_model_short_name(model_version)

                if sum([k in model_short_name for k in skip_model_keys]) > 0:
                    continue

                if (emb_dir / f"{dataset_name}_{model_short_name}.npz").exists() and not args.force_recompute:
                    # Skip if embedding already exists
                    # print(f"Embedding for {dataset_name} with {model_name} {model_version} already exists.")
                    continue

                print(f"Submitting job for {dataset_name} with {model_name} {model_version}")

                if model_name == "Evo2" or model_version == "borzoi" or model_version == "enformer-official-rough":
                    if dataset_name in DATASET_CHUNKINGS:
                        d_num_chunks = DATASET_CHUNKINGS[dataset_name]

                        for d_chunk_ind in range(d_num_chunks):
                            chunk_path = Path(get_embedding_filepath(
                                d.embedding_dir,
                                model_short_name,
                                dataset_name,
                                d_chunk_ind,
                                d_num_chunks,
                            ) + ".npz")
                            if chunk_path.exists() and not args.force_recompute:
                                continue

                            make_and_run_command(
                                model_name,
                                model_version,
                                dataset_name,
                                d_chunk_ind,
                                d_num_chunks,
                                args.batch_size,
                                dry_run=args.dry_run,
                                force_recompute=args.force_recompute
                            )

                        continue

                make_and_run_command(
                    model_name,
                    model_version,
                    dataset_name,
                    0,
                    0,
                    args.batch_size,
                    dry_run=args.dry_run,
                    force_recompute=args.force_recompute
                )
