def get_embedding_filepath(
    output_dir: str,
    model_short_name: str,
    dataset_name: str,
    d_chunk_ind: int = 0,
    d_num_chunks: int = 0
) -> str:
    """Get standardized embedding file path.

    NOTE: Model and dataset names should not have underscores.

    Args:
        output_dir: Directory to store embeddings.
        model_short_name: Short name of model used to generate embeddings.
        dataset_name: Dataset which is embedded.
        d_chunk_ind: Index of current dataset chunk.
        d_num_chunks: Maximum number of dataset chunks.

    Returns:
        Standardized path to embedding file.
    """
    out_fn = get_embedding_filename(
        model_short_name,
        dataset_name,
        d_chunk_ind,
        d_num_chunks
    )
    out_path = "{}/{}".format(output_dir, out_fn)

    return out_path


def get_embedding_filename(
    model_short_name: str,
    dataset_name: str,
    d_chunk_ind: int = 0,
    d_num_chunks: int = 0
) -> str:
    """Get standardized embedding file name.

    NOTE: Model and dataset names should not have underscores.

    Args:
        model_short_name: Short name of model used to generate embeddings.
        dataset_name: Dataset which is embedded.
        d_chunk_ind: Index of current dataset chunk.
        d_num_chunks: Maximum number of dataset chunks.

    Returns:
        Standardized name for embedding file.
    """
    out_fn = "{}_{}".format(dataset_name, model_short_name)

    if d_num_chunks != 0:
        out_fn += "_{}-{}".format(d_chunk_ind, d_num_chunks)

    return out_fn
