import argparse
from pathlib import Path

import torch

from mrna_bench import load_dataset
from orthrus_bench_model import Orthrus
from mrna_bench.embedder import DatasetEmbedder, get_embedding_filepath

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Orthrus", args.model_version, args.checkpoint, args.dataset_name)

    model = Orthrus(
        model_version = args.model_version,
        checkpoint = args.checkpoint,
        device = device,
        model_repository = args.model_dir,
        attn_implementation = None,
    )

    dataset = load_dataset(
        dataset_name=args.dataset_name,
    )

    out_fn_full = get_embedding_filepath(
        dataset.embedding_dir,
        model.short_name,
        dataset.dataset_name
    ) + ".npz"

    if Path(out_fn_full).exists() and not args.force_recompute:
        print(f"Full embedding already computed: {out_fn_full}")
    else:

        embedder = DatasetEmbedder(
            model=model,
            dataset=dataset,
            d_chunk_ind=0,
            d_num_chunks=0
        )

        embedder.persist_embeddings(embedder.embed_dataset())

        print(f"Full embedding computed: {out_fn_full}")
