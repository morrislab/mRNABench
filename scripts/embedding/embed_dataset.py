import argparse
from pathlib import Path

import torch

from mrna_bench import load_model, load_dataset
from mrna_bench.embedder import DatasetEmbedder, get_embedding_filepath


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--d_chunk_ind", type=int, default=0)
parser.add_argument("--d_num_chunks", type=int, default=0)
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":

    print(args.model_name, args.model_version, args.dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        model_name=args.model_name,
        model_version=args.model_version,
        device=device
    )

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        force_redownload=args.force_recompute
    )

    out_fn = get_embedding_filepath(
        dataset.embedding_dir,
        model.short_name,
        dataset.dataset_name,
        d_chunk_ind=args.d_chunk_ind,
        d_num_chunks=args.d_num_chunks
    ) + ".npz"

    out_fn_full = get_embedding_filepath(
        dataset.embedding_dir,
        model.short_name,
        dataset.dataset_name
    ) + ".npz"

    embedder = DatasetEmbedder(
        model=model,
        dataset=dataset,
        d_chunk_ind=args.d_chunk_ind,
        d_num_chunks=args.d_num_chunks
    )

    if Path(out_fn_full).exists() and not args.force_recompute:
        print(f"Full embedding already computed: {out_fn_full}")
        exit(0)
    else:

        if Path(out_fn).exists() and not args.force_recompute:
            print(f"Embedding already computed: {out_fn}")
        else:
            embeddings = embedder.embed_dataset()
            embedder.persist_embeddings(embeddings)

        if args.d_num_chunks > 0:
            embedder.merge_embeddings()
            print(f"Embeddings saved to: {out_fn_full}")
        else:
            print(f"Embeddings saved to: {out_fn}")
