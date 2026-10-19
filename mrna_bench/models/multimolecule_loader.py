"""Checkpoint loading for multimolecule-backed models.

from_pretrained maps checkpoint keys onto a model using the class's
base_model_prefix. Checkpoints published by newer multimolecule releases
store weights under a different prefix than the pinned release declares,
so no key matches, every parameter is randomly initialized, and the load
reports success. Resolve the prefix from the checkpoint itself and
refuse to return a model whose parameters were not populated.
"""

import torch

CHECKPOINT_FILENAMES = ("model.safetensors", "pytorch_model.bin")

# Classification-style head that transformers adds but published
# checkpoints omit. Wrappers embed from last_hidden_state and never
# call it, so leaving it uninitialized does not affect embeddings.
UNUSED_PARAMETER_PREFIXES = ("pooler.",)


def _download_checkpoint(repo_id: str, cache_dir: str) -> dict:
    """Fetch the checkpoint tensors for a repo, using the local cache.

    Args:
        repo_id: HuggingFace repo holding the checkpoint.
        cache_dir: Directory used to cache model weights.

    Returns:
        Mapping of checkpoint key to tensor.

    Raises:
        FileNotFoundError: If the repo exposes no known checkpoint file.
    """
    from huggingface_hub import hf_hub_download

    for filename in CHECKPOINT_FILENAMES:
        try:
            path = hf_hub_download(repo_id, filename, cache_dir=cache_dir)
        except Exception:  # absent file, offline cache miss, or HTTP error
            continue

        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)

        return torch.load(path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        "No checkpoint among {} found for {}.".format(
            CHECKPOINT_FILENAMES, repo_id
        )
    )


def resolve_prefix(checkpoint: dict, target_keys: set[str]) -> str:
    """Find the checkpoint key prefix that maps onto the model's keys.

    Args:
        checkpoint: Mapping of checkpoint key to tensor.
        target_keys: Keys of the model's state dict.

    Returns:
        Prefix to strip, empty when keys already align.
    """
    candidates = {""}
    candidates.update(
        key.split(".")[0] + "." for key in checkpoint if "." in key
    )

    best_prefix = ""
    best_overlap = -1

    for prefix in sorted(candidates):
        stripped = {
            key[len(prefix):] for key in checkpoint if key.startswith(prefix)
        }
        overlap = len(stripped & target_keys)

        if overlap > best_overlap:
            best_prefix = prefix
            best_overlap = overlap

    return best_prefix


def load_multimolecule_model(
    model_cls,
    repo_id: str,
    cache_dir: str,
    unused_prefixes: tuple[str, ...] = UNUSED_PARAMETER_PREFIXES,
):
    """Load a multimolecule model with its pretrained weights.

    Args:
        model_cls: multimolecule model class to instantiate.
        repo_id: HuggingFace repo holding the checkpoint.
        cache_dir: Directory used to cache model weights.
        unused_prefixes: Parameter prefixes exempt from the check,
            for heads the embedding path never calls.

    Returns:
        Model with every embedding parameter populated from the
        checkpoint.

    Raises:
        RuntimeError: If any parameter the embedding path uses is
            absent from the checkpoint, which would leave it randomly
            initialized.
    """
    model = model_cls.from_pretrained(repo_id, cache_dir=cache_dir)

    checkpoint = _download_checkpoint(repo_id, cache_dir)
    prefix = resolve_prefix(checkpoint, set(model.state_dict()))

    weights = {
        key[len(prefix):]: value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }

    unpopulated = {
        name for name in dict(model.named_parameters())
        if name not in weights
        and not name.startswith(unused_prefixes)
    }
    if unpopulated:
        raise RuntimeError(
            "{} parameters of {} are absent from the checkpoint and would "
            "stay randomly initialized (e.g. {}). Resolved prefix {!r} "
            "matched {} of {} checkpoint keys.".format(
                len(unpopulated),
                repo_id,
                sorted(unpopulated)[:3],
                prefix,
                len(weights),
                len(checkpoint),
            )
        )

    # Buffers absent from the checkpoint (e.g. rotary inv_freq) are
    # recomputed from config, so a partial load is expected here.
    model.load_state_dict(weights, strict=False)

    return model
