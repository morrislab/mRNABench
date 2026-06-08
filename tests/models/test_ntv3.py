import pytest

from unittest.mock import patch

pytest.importorskip("torch")

import torch
from mrna_bench.models.nucleotide_transformer_v3 import NucleotideTransformerV3


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def model(device) -> NucleotideTransformerV3:
    """Get NucleotideTransformerV3 model."""
    m = NucleotideTransformerV3("v3_8M_pre", device, "eager")
    m.set_inference_mode()
    return m


def test_ntv3_forward(model):
    """Test NucleotideTransformerV3 initialization and forward pass."""
    out = model.embed_sequence("ATGATG")
    assert out.shape == (1, 256)


def test_ntv3_forward_posttrained(device):
    """Test NucleotideTransformerV3 post-trained model forward pass."""
    m = NucleotideTransformerV3("v3_100M_post", device, "eager")
    m.set_species("human")

    out = m.embed_sequence("ATGATG")
    assert out.shape == (1, 768)


def test_ntv3_forward_non_128(model):
    """Test NucleotideTransformerV3 forward pass with non-multiple of 128 length."""
    long_sequence = "ATGC" * 33  # 132 nucleotides
    out = model.embed_sequence(long_sequence)
    assert out.shape == (1, 256)


def test_ntv3_embed_batch(model):
    """Test NucleotideTransformerV3 batch embedding."""
    sequences = ["ATGATG", "GCGCGC", "AAACCC"]
    out = torch.stack(model.embed(sequences))
    assert out.shape == (3, 256)


@torch.no_grad()
def test_ntv3_embed_batch_ragged(model):
    """Test ragged batches match individual embeddings."""
    sequences = [
        "ACTG" * 5,
        "ACTG" * 50,
        "ACTG" * 33,
        "ACTG" * 10,
    ]

    batch_out = torch.stack(model.embed(sequences)).cpu()
    assert batch_out.shape == (4, 256)

    for i, seq in enumerate(sequences):
        single_out = torch.stack(model.embed([seq])).cpu()
        assert torch.allclose(
            batch_out[i:i + 1], single_out, atol=1e-5
        ), f"Mismatch at sequence {i} (len {len(seq)})"


def test_ntv3_excludes_special_tokens(model):
    """Verify pooling mask reflects no CLS/SEP — only padding is excluded."""
    with patch.object(model, "model") as mock_model:
        seq_len = 6
        # NTv3 pads to multiple of 128; mock returns hidden states accordingly
        mock_model.return_value.hidden_states = [
            torch.ones(1, 128, 256, device=model.device)
        ]

        _, mask = model._forward_chunks(["ATGATG"])

        # Nucleotide positions should be 1 (no CLS/SEP exclusion)
        assert mask[0, :seq_len].sum().item() == seq_len
        # Padding positions (beyond seq_len, up to 128) should be 0
        assert mask[0, seq_len:].sum().item() == 0


def test_ntv3_posttrained_requires_species(device):
    """Test that post-trained model auto-sets species when none given."""
    import warnings
    m = NucleotideTransformerV3("v3_100M_post", device, "eager")
    m.set_inference_mode()

    # species_id should be None before any embed call
    assert m.species_id is None

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = m.embed(["ATGATG"])
        assert any("species" in str(warning.message).lower() for warning in w)

    assert out[0].shape == (768,)


@torch.no_grad()
def test_ntv3_embed_ragged_agg(model):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = model.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_ntv3_gradient_flow(model):
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

NTV3_EXTRACT_SEQ = "ATGC" * 40  # 160 nt — min required by 7× avg_pool


def test_ntv3_extract_structure(model):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = model.extract([NTV3_EXTRACT_SEQ], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_ntv3_extract_layer_selection(model):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = model.extract([NTV3_EXTRACT_SEQ], layers=[0])
    assert len(h) == 1


def test_ntv3_extract_conv_tower_downsampling(model):
    """Conv tower layers produce progressively shorter sequences (stride > 1)."""
    h, _ = model.extract([NTV3_EXTRACT_SEQ])
    T_first = h["core.conv_tower_blocks.0"][0][0].shape[0]
    T_last_conv = h["core.conv_tower_blocks.6"][0][0].shape[0]
    assert T_last_conv < T_first


def test_ntv3_extract_transformer_attention(model):
    """Transformer blocks expose (H, T, T) attention weights with eager."""
    _, s = model.extract(
        [NTV3_EXTRACT_SEQ],
        layers=["core.transformer_blocks.0"],
        return_attentions=True,
    )
    attn = s["core.transformer_blocks.0"]
    assert attn is not None
    w = attn[0][0]  # (H, T, T)
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]


def test_ntv3_extract_conv_deconv_scores_none(model):
    """Conv and deconv tower layers return None scores (no attention matrix)."""
    _, s = model.extract(
        [NTV3_EXTRACT_SEQ],
        layers=["core.conv_tower_blocks.0", "core.deconv_tower_blocks.0"],
        return_attentions=True,
    )
    assert s["core.conv_tower_blocks.0"] is None
    assert s["core.deconv_tower_blocks.0"] is None
