def get_output_filename(
    output_dir: str,
    model_version: str,
    dataset_name: str,
    sequence_chunk_overlap: int,
    d_chunk_ind: int = 0,
    d_num_chunks: int = 0
) -> str:
    """Get standardized embedding file name.

    NOTE: Model and dataset names should not have underscores.

    Args:
        output_dir: Directory to store embeddings.
        model_version: Name of embedding model version.
        dataset_name: Dataset which is embedded.
        sequence_chunk_overlap: Number of tokens overlapped in sequence chunks.
        d_chunk_ind: Index of current dataset chunk.
        d_num_chunks: Maximum number of dataset chunks.
    """
    out_path = "{}/{}_{}_o{}".format(
        output_dir,
        dataset_name,
        model_version,
        sequence_chunk_overlap
    )

    if d_num_chunks != 0:
        out_path += "_{}-{}".format(d_chunk_ind, d_num_chunks)

    return out_path
