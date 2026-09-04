import pytest

pytest.importorskip("torch")
import torch

from tests.model_utils import (
    assert_pooled_batch_matches_single,
    assert_raw_batch_matches_single,
)

from mrna_bench.models.rnamsm import RNAMSM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> RNAMSM:
    """Get RNA-MSM model."""
    return RNAMSM("RNA-MSM", device, "eager")


def test_rnamsm_forward(model):
    """Test RNA-MSM initialization and forward pass."""
    model.set_inference_mode()
    text = "ACUUGGCCA"
    output = torch.stack(model.embed([text])).cpu()
    assert output.shape == (1, 768)


def test_rnamsm_excludes_only_cls(model):
    """RNA-MSM prepends CLS but appends no EOS.

    Pooling should drop only the leading CLS and keep every real nucleotide,
    so the pooled-token count equals the sequence length.
    """
    model.set_inference_mode()
    seq = "ACGUACGU"  # 8 nucleotides
    _, pooling_mask = model._forward_chunks([seq])

    # CLS lives at index 0 and must be excluded.
    assert int(pooling_mask[0, 0].item()) == 0
    # All real nucleotides are kept (no trailing EOS to drop).
    assert int(pooling_mask.sum().item()) == len(seq)


def test_rnamsm_forward_dna_input(model):
    """Test RNA-MSM forward pass with DNA input (T instead of U)."""
    model.set_inference_mode()
    text_rna = "ACUUGGCCA"
    text_dna = "ACTTGGCCA"

    output_rna = torch.stack(model.embed([text_rna])).cpu()
    output_dna = torch.stack(model.embed([text_dna])).cpu()

    assert torch.equal(output_rna, output_dna)


def test_rnamsm_embed_batch(model):
    """Test batch embed matches individual embeddings."""
    model.set_inference_mode()
    sequences = [
        "ACUUGGCCA",
        "GGCCAAUUGG",
        "UUUAAAGGGCCC",
    ]

    assert_pooled_batch_matches_single(model, sequences)


@torch.no_grad()
def test_rnamsm_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert_raw_batch_matches_single(model, seqs, out)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_rnamsm_gradient_flow(model):
    """Test that gradients can flow through the model."""
    model.set_train_mode()

    out = model.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in model.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    model.set_inference_mode()


def test_rnamsm_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_rnamsm_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_rnamsm_extract_attention_weights(model):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = model.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
