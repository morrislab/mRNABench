import gc

import pytest


def pytest_collection_modifyitems(items: list) -> None:
    """Move evo2 tests to the end of the session.

    Importing evo2 imports TransformerEngine, whose module-level
    initialisation rewrites PyTorch's CUDA op dispatcher and corrupts the
    flash_attn schema used by OmniGenome and Evo1. By deferring evo2 until
    after all other model tests we avoid this cross-contamination.
    """
    evo2_items = [i for i in items if "test_evo2" in i.nodeid]
    other_items = [i for i in items if "test_evo2" not in i.nodeid]
    items[:] = other_items + evo2_items


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
