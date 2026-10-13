from pathlib import Path
import hashlib
import json

import h5py
import numpy as np

from mrna_bench.models import EmbeddingModel
from mrna_bench.models.pooling import MeanPool, Pooling


def get_embedding_config(
    model: EmbeddingModel | type[EmbeddingModel],
    pooling: Pooling = MeanPool(),
    layer: str = "final",
) -> dict:
    """Get the settings that determine a model's persisted embeddings.

    Args:
        model: Model used to generate embeddings, or its class when it is
            loaded with the default attention implementation.
        pooling: Pooling applied to the token states.
        layer: Layer whose token states are pooled.

    Returns:
        Embedding config stored with the embeddings and hashed into their
        file name.
    """
    attn_implementation = getattr(model, "attn_implementation", None)
    config = {
        "pooling": pooling.config,
        "layer": layer,
        "attn_implementation": (
            attn_implementation or model.default_attn_implementation
        ),
    }
    if model.model_options:
        config["model_options"] = dict(model.model_options)
    return config


def get_config_hash(config: dict) -> str:
    """Get the short hash of an embedding config used in file names.

    Args:
        config: Embedding config.

    Returns:
        First 8 hex digits of the SHA-1 of the config.
    """
    # TODO: adding or renaming a config key changes every hash; warn users
    # and rename existing embeddings using the config stored in each file.
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha1(config_str.encode()).hexdigest()[:8]


def read_embedding_config(path: str | Path) -> dict:
    """Read the embedding config stored in an .npz or .h5 embedding file.

    Args:
        path: Path to embedding file.

    Returns:
        Embedding config.
    """
    if Path(path).suffix == ".h5":
        with h5py.File(path, "r") as f:
            return json.loads(f.attrs["config"])
    else:
        with np.load(path) as f:
            return json.loads(str(f["config"]))


def find_embedding_file(
    output_dir: str,
    model_short_name: str,
    dataset_name: str,
    pooling: Pooling = MeanPool(),
    layer: str = "final",
    attn_implementation: str | None = None,
    model_options: dict | None = None,
) -> Path:
    """Find the embedding file whose stored config matches.

    Args:
        output_dir: Directory where embeddings are stored.
        model_short_name: Short name of model used to generate embeddings.
        dataset_name: Dataset which is embedded.
        pooling: Pooling applied to the token states.
        layer: Layer whose token states are pooled.
        attn_implementation: Attention implementation. None accepts any, as
            long as only one file matches.
        model_options: Model options that differ from their defaults, see
            EmbeddingModel.model_options. None means all defaults.

    Returns:
        Path to the matching embedding file.

    Raises:
        FileNotFoundError: If no file matches.
        ValueError: If several files match.
    """
    wanted = {
        "pooling": pooling.config,
        "layer": layer,
        "model_options": model_options or {},
    }
    if attn_implementation is not None:
        wanted["attn_implementation"] = attn_implementation

    matches = []
    pattern = "{}_{}_*".format(dataset_name, model_short_name)
    for path in Path(output_dir).glob(pattern):
        # skip unmerged chunks
        if path.suffix not in (".npz", ".h5") or path.stem.count("_") != 2:
            continue
        config = {"model_options": {}, **read_embedding_config(path)}
        if all(config.get(k) == v for k, v in wanted.items()):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "No {} embeddings of {} with {}.".format(
                model_short_name, dataset_name, wanted
            )
        )
    if len(matches) > 1:
        raise ValueError(
            "Several embedding files match, set attn_implementation: "
            "{}".format(sorted(p.name for p in matches))
        )
    return matches[0]


def get_embedding_filepath(
    output_dir: str,
    model_short_name: str,
    dataset_name: str,
    config_hash: str,
    d_chunk_ind: int = 0,
    d_num_chunks: int = 0
) -> str:
    """Get standardized embedding file path.

    NOTE: Model and dataset names should not have underscores.

    Args:
        output_dir: Directory to store embeddings.
        model_short_name: Short name of model used to generate embeddings.
        dataset_name: Dataset which is embedded.
        config_hash: Hash of the embedding config.
        d_chunk_ind: Index of current dataset chunk.
        d_num_chunks: Maximum number of dataset chunks.

    Returns:
        Standardized path to embedding file.
    """
    out_fn = get_embedding_filename(
        model_short_name,
        dataset_name,
        config_hash,
        d_chunk_ind,
        d_num_chunks
    )
    out_path = "{}/{}".format(output_dir, out_fn)

    return out_path


def get_embedding_filename(
    model_short_name: str,
    dataset_name: str,
    config_hash: str,
    d_chunk_ind: int = 0,
    d_num_chunks: int = 0
) -> str:
    """Get standardized embedding file name.

    NOTE: Model and dataset names should not have underscores.

    Args:
        model_short_name: Short name of model used to generate embeddings.
        dataset_name: Dataset which is embedded.
        config_hash: Hash of the embedding config.
        d_chunk_ind: Index of current dataset chunk.
        d_num_chunks: Maximum number of dataset chunks.

    Returns:
        Standardized name for embedding file.
    """
    out_fn = "{}_{}_{}".format(dataset_name, model_short_name, config_hash)

    if d_num_chunks != 0:
        out_fn += "_{}-{}".format(d_chunk_ind, d_num_chunks)

    return out_fn
