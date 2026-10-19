import pytest

import torch
import torch.nn as nn

from mrna_bench.fine_tune.fine_tune_wrapper import FineTuneWrapper
from mrna_bench.fine_tune.poolers import AttentionPooler
from mrna_bench.fine_tune.task_heads import TaskHead
from mrna_bench.fine_tune.trainer import FineTuneTrainer, TrainerConfig
from mrna_bench.models.embedding_model import EmbeddingModel


HIDDEN_DIM = 16


class StubModel(EmbeddingModel):
    """Backbone emitting fixed token embeddings, pooled by agg_fn."""

    default_version = "stub"
    valid_versions = ["stub"]

    def __init__(self, model_version="stub", device=torch.device("cpu")):
        """Initialize StubModel with a trainable linear layer."""
        super().__init__(model_version, device)
        self.model = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def embed(self, sequences, cds=None, splice=None, agg_fn=torch.mean):
        """Emit per-sequence token embeddings pooled over the token axis."""
        pooled = []
        for seq in sequences:
            tokens = torch.ones(len(seq), HIDDEN_DIM)
            pooled.append(agg_fn(self.model(tokens), dim=0))
        return torch.stack(pooled, dim=0)


@pytest.fixture
def pooler():
    """Attention pooler over HIDDEN_DIM embeddings."""
    torch.manual_seed(0)
    return AttentionPooler(HIDDEN_DIM, num_heads=4)


@pytest.fixture
def wrapper(pooler):
    """FineTuneWrapper with a stub backbone and attention pooler."""
    head = TaskHead(HIDDEN_DIM, output_dim=1, task_type="regression")
    return FineTuneWrapper(StubModel(), head, pooler=pooler)


def test_rejects_indivisible_head_count():
    """Check num_heads must divide hidden_dim."""
    with pytest.raises(ValueError, match="must divide hidden_dim"):
        AttentionPooler(HIDDEN_DIM, num_heads=5)


def test_rejects_hidden_dim_mismatch(pooler):
    """Check a wrong hidden axis is rejected rather than silently pooled."""
    with pytest.raises(ValueError, match="Expected hidden dim"):
        pooler(torch.randn(8, HIDDEN_DIM + 1), dim=0)


@pytest.mark.parametrize("shape,dim", [
    ((8, HIDDEN_DIM), 0),                # chunked wrappers: (len, hidden)
    ((4, 8, HIDDEN_DIM), 1),             # helix_mrna/evo2: (batch, len, hid)
    ((4, HIDDEN_DIM, 8), 2),             # borzoi/enformer: (batch, hid, len)
])
def test_output_shape_matches_mean(pooler, shape, dim):
    """Check pooled shape matches torch.mean for every wrapper convention."""
    x = torch.randn(*shape)

    pooled = pooler(x, dim=dim)

    assert pooled.shape == torch.mean(x, dim=dim).shape


def test_attends_non_uniformly(pooler):
    """Check pooling is token-weighted rather than an unweighted mean."""
    x = torch.randn(8, HIDDEN_DIM)

    assert not torch.allclose(pooler(x, dim=0), torch.mean(x, dim=0))


def test_casts_input_to_parameter_dtype(pooler):
    """Check a half precision backbone output pools to the head's dtype."""
    pooled = pooler(torch.randn(8, HIDDEN_DIM, dtype=torch.bfloat16), dim=0)

    assert pooled.dtype == torch.float32


def test_gradients_reach_pooler(pooler):
    """Check pooled output backpropagates into the pooler parameters."""
    pooler(torch.randn(8, HIDDEN_DIM), dim=0).sum().backward()

    assert pooler.query.grad is not None
    assert not torch.allclose(
        pooler.query.grad, torch.zeros_like(pooler.query)
    )


def test_wrapper_registers_pooler_parameters(wrapper, pooler):
    """Check pooler parameters are handed to the optimizer."""
    trainable = wrapper.get_trainable_parameters()

    for param in pooler.parameters():
        assert any(param is p for p in trainable)


def test_wrapper_counts_pooler_parameters(wrapper, pooler):
    """Check pooler parameters are included in the reported totals."""
    counts = wrapper.get_parameter_count()
    expected = sum(p.numel() for p in pooler.parameters())

    assert counts["pooler"] == expected
    assert counts["total_trainable"] == (
        counts["backbone_trainable"] + counts["head"] + expected
    )


def test_wrapper_defaults_to_mean_without_pooler():
    """Check the wrapper still mean pools when no pooler is given."""
    head = TaskHead(HIDDEN_DIM, output_dim=1, task_type="regression")
    wrapper = FineTuneWrapper(StubModel(), head)

    assert wrapper.pooler is None
    assert wrapper.get_parameter_count()["pooler"] == 0
    assert wrapper.forward(["ACGT", "ACG"]).shape == (2,)


def test_wrapper_forward_uses_pooler(wrapper):
    """Check forward routes the pooler into the backbone embed call."""
    assert wrapper.forward(["ACGT", "ACG"]).shape == (2,)


def test_trainer_round_trips_pooler_state(wrapper, pooler):
    """Check best-model restore does not drop the pooler weights."""
    trainer = FineTuneTrainer(wrapper, TrainerConfig())

    state = trainer._save_trainable_state()
    with torch.no_grad():
        pooler.query.add_(1.0)
    trainer._restore_trainable_state(state)

    assert torch.allclose(pooler.query, state["pooler"]["query"])


def test_freeze_backbone_leaves_only_pooler_and_head(wrapper, pooler):
    """Check freezing trains the pooler and head but not the backbone."""
    wrapper.freeze_backbone()

    counts = wrapper.get_parameter_count()
    expected = sum(p.numel() for p in pooler.parameters())

    assert counts["backbone_trainable"] == 0
    assert counts["total_trainable"] == counts["head"] + expected
    assert all(
        not p.requires_grad for p in wrapper.backbone.model.parameters()
    )


def test_freeze_backbone_keeps_backbone_in_eval(wrapper):
    """Check training mode does not flip a frozen backbone into train."""
    wrapper.freeze_backbone()
    wrapper.set_train_mode()

    assert not wrapper.backbone.model.training
    assert wrapper.task_head.training
    assert wrapper.pooler.training


def test_frozen_backbone_trains_pooler(wrapper, pooler):
    """Check gradients reach the pooler through a frozen backbone."""
    wrapper.freeze_backbone()
    wrapper.set_train_mode()

    wrapper.forward(["ACGT", "ACG"]).sum().backward()

    assert pooler.query.grad is not None
    assert all(
        p.grad is None for p in wrapper.backbone.model.parameters()
    )


def test_frozen_backbone_saves_state_without_lora(wrapper, pooler):
    """Check best-state save works when no LoRA adapters were applied."""
    wrapper.freeze_backbone()
    trainer = FineTuneTrainer(wrapper, TrainerConfig())

    state = trainer._save_trainable_state()

    assert "lora" not in state
    assert "pooler" in state

    with torch.no_grad():
        pooler.query.add_(1.0)
    trainer._restore_trainable_state(state)

    assert torch.allclose(pooler.query, state["pooler"]["query"])


def test_freeze_and_lora_are_mutually_exclusive(wrapper):
    """Check freezing after LoRA (and vice versa) is refused."""
    wrapper.freeze_backbone()

    with pytest.raises(RuntimeError, match="Backbone is frozen"):
        wrapper.apply_lora()

    wrapper.backbone_frozen = False
    wrapper.lora_applied = True

    with pytest.raises(RuntimeError, match="LoRA adapters are applied"):
        wrapper.freeze_backbone()
