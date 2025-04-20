from typing import TYPE_CHECKING, Type

from mrna_bench.datasets import BenchmarkDataset, DATASET_CATALOG

if TYPE_CHECKING:
    import torch
    from mrna_bench.models import EmbeddingModel

# TODO:
# Remove references to loading custom Orthrus model checkpoints
# and replace with loading from HuggingFace hub.
# and kwargs for loading datasets

def load_model(
    model_name: str,
    model_version: str,
    device: "torch.device",
    checkpoint: str = None
) -> "EmbeddingModel":
    """Load Embedding Model.

    Args:
        model_name: Name of model class.
        model_version: Specific model version to load.
        device: PyTorch device to load model to.

    Returns:
        Initialized EmbeddingModel.
    """
    try:
        from mrna_bench.models import EmbeddingModel, MODEL_CATALOG

        model_class: Type[EmbeddingModel] = MODEL_CATALOG[model_name]
        
        if model_name != "Orthrus" or checkpoint is None:
            model = model_class(model_version, device)
        else:
            model = model_class(model_version, checkpoint, device)

    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "PyTorch not installed. Model benchmarking unavailable."
        )

    return model


def load_dataset(
    dataset_name: str,
    force_redownload: bool = False,
    **kwargs # noqa
) -> BenchmarkDataset:
    """Load Benchmark Dataset.

    Args:
        dataset_name: Name of the dataset.
        force_redownload: Forces data file redownload.
    """

    return DATASET_CATALOG[dataset_name](force_redownload, **kwargs)
