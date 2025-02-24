import argparse
from pathlib import Path

import torch

from mrna_bench import load_model, load_dataset
from mrna_bench.embedder import DatasetEmbedder, get_output_filepath

parser = argparse.ArgumentParser()
parser.add_argument("--model_class", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--isoform_resolved", type=str)
parser.add_argument("--target_col", type=str)
parser.add_argument("--transcript_avg", type=str)
parser.add_argument("--s_chunk_overlap", type=int, default=0)
parser.add_argument("--d_chunk_ind", type=int, default=0)
parser.add_argument("--d_num_chunks", type=int, default=0)
parser.add_argument("--force_recompute", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    isoform_resolved = True if args.isoform_resolved == "True" else False
    transcript_avg = True if args.transcript_avg == "True" else False

    model = load_model(
        model_name = args.model_class, 
        model_version = args.model_version, 
        device = device)

    dataset = load_dataset(
        dataset_name = args.dataset_name, 
        isoform_resolved = isoform_resolved,
        target_col = args.target_col,
        force_redownload = args.force_recompute)

    out_fn = get_output_filepath(
        dataset.embedding_dir,
        model.short_name,
        dataset.dataset_name,
        args.s_chunk_overlap,
        d_chunk_ind=args.d_chunk_ind,
        d_num_chunks=args.d_num_chunks
    )

    embedder = DatasetEmbedder(
        model=model,
        dataset=dataset,
        s_chunk_overlap=args.s_chunk_overlap,
        d_chunk_ind=args.d_chunk_ind,
        d_num_chunks=args.d_num_chunks,
        transcript_avg=transcript_avg
    )

    if Path(out_fn + ".npz").exists() and not args.force_recompute:
        print("Embedding already computed.")
    else:
        if transcript_avg:
            embeddings, gene_list, row_index_list = embedder.embed_dataset()
            embedder.persist_embeddings(embeddings, gene_list, row_index_list)
        else:
            embeddings = embedder.embed_dataset()
            embedder.persist_embeddings(embeddings)

    if args.d_num_chunks != 0:
        embedder.merge_embeddings()