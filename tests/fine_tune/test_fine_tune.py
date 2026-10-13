import torch

from mrna_bench.fine_tune import (
    FineTuneWrapper,
    TaskHead,
)
from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior


class TinyBackbone(EmbeddingModel):
    default_version = "tiny"
    valid_versions = ["tiny"]
    default_attn_implementation = None
    valid_attn_implementations = None
    hookable_layer_patterns = []
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    def __init__(self):
        super().__init__("tiny", torch.device("cpu"))
        self.model = torch.nn.Linear(1, 2)

    def embed(self, sequences, cds=None, splice=None, agg_fn=torch.mean):
        values = torch.tensor(
            [[float(len(sequence))] for sequence in sequences]
        )
        return list(self.model(values))


class LayeredBackbone(TinyBackbone):
    """Backbone with hookable layers."""

    hookable_layer_patterns = [r"layers\.\d+"]

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            torch.nn.Linear(2, 2) for _ in range(3)
        )


def test_wrapper_trains_only_requested_layers():
    """No trainable_layers keeps the backbone frozen; [-1] the last layer."""
    frozen = LayeredBackbone()
    FineTuneWrapper(frozen, TaskHead(2, 1, "regression"))
    assert not any(p.requires_grad for p in frozen.model.parameters())

    last = LayeredBackbone()
    FineTuneWrapper(last, TaskHead(2, 1, "regression"), trainable_layers=[-1])
    trainable = [p.requires_grad for p in last.model.parameters()]
    assert trainable == [False, False, False, False, True, True]


def test_head_only_state_round_trips():
    """Without PEFT, backbone is frozen and only head state is saved."""
    backbone = TinyBackbone()
    head = TaskHead(2, 1, "regression")
    wrapper = FineTuneWrapper(backbone, head)

    assert not any(
        p.requires_grad for p in backbone.model.parameters()
    )

    state = wrapper.trainable_state_dict()
    assert all(not s for s in state["lora"])
    assert state["head"]

    original = head.mlp[0].weight.detach().clone()
    with torch.no_grad():
        head.mlp[0].weight.add_(1)
    wrapper.load_trainable_state_dict(state)

    assert torch.equal(head.mlp[0].weight, original)


def test_multiclass_head_metrics():
    """Report macro and micro metrics for multiclass heads."""
    head = TaskHead(2, 3, "classification")
    logits = torch.tensor([
        [4.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [0.0, 4.0, 1.0],
        [0.0, 3.0, 1.0],
        [0.0, 1.0, 4.0],
        [0.0, 1.0, 3.0],
    ])
    targets = torch.tensor([0, 0, 1, 1, 2, 2])

    metrics = head.compute_metrics(logits, targets)

    assert metrics["mcc"] == 1.0
    assert metrics["auprc_macro"] == 1.0
    assert metrics["auprc_micro"] == 1.0


def test_trainable_peft_state_round_trips():
    """Save and restore real PEFT adapter weights."""
    from transformers import BertConfig, BertModel

    backbone = TinyBackbone()
    backbone.model = BertModel(BertConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
    ))
    head = TaskHead(8, 1, "regression")
    wrapper = FineTuneWrapper(backbone, head)
    wrapper.apply_lora(
        rank=2,
        alpha=4,
        target_modules=["query", "value"],
    )

    state = wrapper.trainable_state_dict()
    assert any("lora_" in name for s in state["lora"] for name in s)

    parameter = next(
        parameter
        for parameter in backbone.model.parameters()
        if parameter.requires_grad
    )
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1)
    wrapper.load_trainable_state_dict(state)

    assert torch.equal(parameter, original)
