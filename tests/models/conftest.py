import gc

import pytest


@pytest.fixture(autouse=True)
def batch_tests_use_inference_mode(request):
    """Batch comparisons mirror DatasetEmbedder inference."""
    import torch
    if "embed_batch" in request.node.name:
        with torch.inference_mode():
            yield
    else:
        yield


@pytest.fixture(autouse=True, scope="module")
def free_gpu_memory_between_modules():
    """Force GPU memory release after every test module.

    Module-scoped model fixtures stay alive for the entire module, then
    are torn down at module exit.  Python's GC may not immediately
    collect the PyTorch tensors, leaving CUDA memory occupied for the
    next module.  Calling gc.collect() + cuda.empty_cache() here
    (after all other module fixtures have been torn down) tries to ensure
    that memory is freed before the next model is loaded.
    """
    yield
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
