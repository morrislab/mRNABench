import sys
import importlib

def pytest_runtest_teardown(item):
    # Drop flash-attn state between tests
    if "flash_attn" in sys.modules:
        del sys.modules["flash_attn"]
        del sys.modules["flash_attn.flash_attn_interface"]
        # force re-import cleanly next time
        try:
            importlib.invalidate_caches()
        except Exception:
            pass
