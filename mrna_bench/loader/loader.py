from typing import TYPE_CHECKING, Type

from mrna_bench.datasets import BenchmarkDataset, DATASET_CATALOG

if TYPE_CHECKING:
    import torch
    from mrna_bench.models import EmbeddingModel


def load_model(
    model_name: str,
    model_version: str | None = None,
    device: "torch.device | None" = None,
    attn_implementation: str | None = None,
) -> "EmbeddingModel":
    """Load Embedding Model.

    Args:
        model_name: Name of model class.
        model_version: Specific model version to load. Defaults to model's
            default_version if not specified.
        device: PyTorch device to load model to. Defaults to CUDA if available.
        attn_implementation: Attention implementation override.  One of
            "eager", "sdpa", or "flash_attention_2".  Pass None to use the
            model's default.

    Returns:
        Initialized EmbeddingModel in inference mode.
    """
    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyTorch not installed. Model benchmarking unavailable."
        ) from None

    from mrna_bench.models import EmbeddingModel, MODEL_CATALOG

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model_class: Type[EmbeddingModel] = MODEL_CATALOG[model_name]

    if model_version is None:
        model_version = model_class.default_version

    if attn_implementation is None:
        attn_implementation = model_class.default_attn_implementation

    model = model_class(
        model_version, device, attn_implementation=attn_implementation
    )
    model.set_inference_mode()

    return model


def load_dataset(
    dataset_name: str,
    force_redownload_hf: bool = False,
    force_rebuild_raw: bool = False
) -> BenchmarkDataset:
    """Load Benchmark Dataset.

    Args:
        dataset_name: Name of the dataset.
        force_redownload_hf: Forces redownload from HuggingFace.
        force_rebuild_raw: Forces rebuild from raw data source.
    """
    return DATASET_CATALOG[dataset_name](
        force_redownload_hf=force_redownload_hf,
        force_rebuild_raw=force_rebuild_raw
    )
