import torch

from mrna_bench.datasets import BenchmarkDataset, DATASET_CATALOG
from mrna_bench.models import EmbeddingModel, MODEL_CATALOG


def load_model(
    model_name: str,
    model_version: str,
    device: torch.device
) -> EmbeddingModel:
    """Load Embedding Model.

    Args:
        model_name: Name of model class.
        model_version: Specific model version to load.
        device: PyTorch device to load model to.

    Returns:
        Initialized EmbeddingModel.
    """
    try:
        model_class: EmbeddingModel = MODEL_CATALOG[model_name]
        model = model_class(model_version, device)
    except ModuleNotFoundError:
        print("Error loading model dependencies. Check environment.")
        raise

    return model


def load_dataset(
    dataset_name: str,
    force_redownload: bool = False
) -> BenchmarkDataset:
    """Load Benchmark Dataset.

    Args:
        dataset_name: Name of the dataset.
        force_redownload: Forces data file redownload.
    """
    return DATASET_CATALOG[dataset_name](force_redownload)
