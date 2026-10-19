import pytest

import torch
import torch.nn as nn
from unittest.mock import patch

from mrna_bench.models.multimolecule_loader import (
    load_multimolecule_model,
    resolve_prefix,
)


class StubModel(nn.Module):
    """Two-parameter stand-in for a multimolecule encoder."""

    def __init__(self):
        """Build an encoder weight plus an unused pooler head."""
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.pooler = nn.Linear(4, 4, bias=False)

    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None):
        """Mimic the transformers constructor."""
        return cls()


def checkpoint_for(prefix: str, include_pooler: bool = False) -> dict:
    """Build a checkpoint whose keys carry the given prefix."""
    weights = {prefix + "encoder.weight": torch.ones(4, 4)}
    if include_pooler:
        weights[prefix + "pooler.weight"] = torch.ones(4, 4)
    return weights


def test_resolves_empty_prefix_when_keys_already_align():
    """Check an aligned checkpoint needs no prefix stripped."""
    ckpt = checkpoint_for("")

    assert resolve_prefix(ckpt, {"encoder.weight"}) == ""


def test_resolves_prefix_used_by_newer_releases():
    """Check the model. prefix that breaks the pinned release is found."""
    ckpt = checkpoint_for("model.")

    assert resolve_prefix(ckpt, {"encoder.weight"}) == "model."


def test_resolves_arbitrary_prefix():
    """Check prefix detection is not hardcoded to a known name."""
    ckpt = checkpoint_for("rinalmo.")

    assert resolve_prefix(ckpt, {"encoder.weight"}) == "rinalmo."


def test_prefers_the_prefix_matching_more_keys():
    """Check the best-overlap prefix wins over a partial match."""
    ckpt = {
        "model.encoder.weight": torch.ones(4, 4),
        "model.pooler.weight": torch.ones(4, 4),
        "other.encoder.weight": torch.zeros(4, 4),
    }

    resolved = resolve_prefix(ckpt, {"encoder.weight", "pooler.weight"})

    assert resolved == "model."


def test_loads_prefixed_checkpoint_into_the_model():
    """Check a prefixed checkpoint populates the encoder weights."""
    with patch(
        "mrna_bench.models.multimolecule_loader._download_checkpoint",
        return_value=checkpoint_for("model."),
    ):
        model = load_multimolecule_model(StubModel, "repo", "cache")

    assert torch.equal(model.encoder.weight, torch.ones(4, 4))


def test_raises_when_no_prefix_matches():
    """Check a wholly unmatched checkpoint fails instead of returning
    a silently random model."""
    ckpt = {"totally.different.weight": torch.ones(4, 4)}

    with patch(
        "mrna_bench.models.multimolecule_loader._download_checkpoint",
        return_value=ckpt,
    ):
        with pytest.raises(RuntimeError, match="randomly initialized"):
            load_multimolecule_model(StubModel, "repo", "cache")


def test_tolerates_absent_pooler_head():
    """Check the unused pooler may be missing from the checkpoint."""
    with patch(
        "mrna_bench.models.multimolecule_loader._download_checkpoint",
        return_value=checkpoint_for("model.", include_pooler=False),
    ):
        model = load_multimolecule_model(StubModel, "repo", "cache")

    assert torch.equal(model.encoder.weight, torch.ones(4, 4))


def test_reports_the_unpopulated_parameter():
    """Check the error names a parameter that would stay random."""
    ckpt = {"model.pooler.weight": torch.ones(4, 4)}

    with patch(
        "mrna_bench.models.multimolecule_loader._download_checkpoint",
        return_value=ckpt,
    ):
        with pytest.raises(RuntimeError, match="encoder.weight"):
            load_multimolecule_model(StubModel, "repo", "cache")
