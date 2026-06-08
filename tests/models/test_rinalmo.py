import pytest

from unittest.mock import patch
from functools import partial

pytest.importorskip("torch")
import torch

from mrna_bench.models.rinalmo import RiNALMo


RINALMO_VERSIONS = [
    ("RiNALMo-giga", 1280),
    ("RiNALMo-mega", 640),
    ("RiNALMo-micro", 480),
]


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module", params=RINALMO_VERSIONS, ids=lambda x: x[0])
def rinalmo(request, device):
    """Get RiNALMo model and expected embedding dim."""
    model_version, embed_dim = request.param
    model = RiNALMo(model_version, device, "eager")
    model.set_inference_mode()
    return model, embed_dim


@pytest.fixture(scope="module")
def rinalmo_giga(device) -> RiNALMo:
    """Get RiNALMo giga model for specific output tests."""
    model = RiNALMo("RiNALMo-giga", device, "eager")
    model.set_inference_mode()
    return model


def test_rinalmo_forward(rinalmo):
    """Test RiNALMo forward pass."""
    model, embed_dim = rinalmo

    text = "ACTTTGGCCA"
    output = model.embed_sequence(
        text,
        agg_fn=partial(torch.mean, dim=0)
    ).cpu()
    assert output.shape == (1, embed_dim)


def test_rinalmo_giga_output(rinalmo_giga):
    """Test RiNALMo giga produces expected output."""
    text = "ACTTTGGCCA"
    output = rinalmo_giga.embed_sequence(
        text,
        agg_fn=partial(torch.mean, dim=0)
    ).cpu()
    assert output.shape == (1, 1280)

    # Matches output from official release
    assert torch.allclose(
        torch.Tensor([-0.00032]),
        torch.mean(output),
        atol=0.0001
    )


def test_rinalmo_converts_t_to_u(rinalmo):
    """Test that RiNALMo forward pass automatically converts T->U."""
    model, _ = rinalmo
    text = "ACTTTGGCCA"
    with patch.object(
        model,
        "tokenizer",
        side_effect=model.tokenizer
    ) as mock:
        model.embed_sequence(text)
        mock.assert_called_once_with(
            ["ACUUUGGCCA"],
            return_tensors="pt",
            padding=True,
        )


def test_rinalmo_embed_batch(rinalmo):
    """Test RiNALMo batch embedding."""
    model, embed_dim = rinalmo
    sequences = ["ACTTTGGCCA", "GGCCAATTGG", "AAAAACCCCC"]
    output = torch.stack(model.embed(sequences)).cpu()
    assert output.shape == (3, embed_dim)


def test_rinalmo_embed_batch_single_equals_embed_sequence(rinalmo):
    """Test that embed_batch with single sequence matches embed_sequence."""
    model, _ = rinalmo
    text = "ACTTTGGCCA"
    single_output = model.embed_sequence(text).cpu()
    batch_output = torch.stack(model.embed([text])).cpu()
    assert torch.allclose(single_output, batch_output, atol=1e-5)


@torch.no_grad()
def test_rinalmo_embed_batch_with_chunking(rinalmo):
    """Test batch embedding with sequences requiring chunking."""
    model, embed_dim = rinalmo
    short_seq = "ACTG" * 100
    long_seq = "ACTG" * 3000

    output = torch.stack(model.embed([short_seq, long_seq])).cpu()
    assert output.shape == (2, embed_dim)

    single_short = model.embed_sequence(short_seq).cpu()
    single_long = model.embed_sequence(long_seq).cpu()

    assert torch.allclose(output[0:1], single_short, atol=1e-5)
    assert torch.allclose(output[1:2], single_long, atol=1e-5)


@torch.no_grad()
def test_rinalmo_embed_batch_ragged(rinalmo):
    """Test ragged batches match individual embeddings."""
    model, embed_dim = rinalmo
    sequences = [
        "ACTG" * 5,
        "ACTG" * 50,
        "ACTG" * 200,
        "ACTG" * 10,
    ]

    batch_output = torch.stack(model.embed(sequences)).cpu()
    assert batch_output.shape == (4, embed_dim)

    for i, seq in enumerate(sequences):
        single_output = model.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-5
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_rinalmo_excludes_special_tokens(rinalmo):
    """Test that CLS and EOS tokens are excluded from pooling."""
    model, embed_dim = rinalmo
    text = "ACUG" * 10

    toks = model.tokenizer([text], return_tensors="pt", padding=True)
    toks = toks.to(model.device)
    hidden_states = model.model(**toks).last_hidden_state

    # Mean over ALL tokens (including CLS/EOS)
    mean_all = hidden_states.mean(dim=1).cpu()

    # Mean excluding first and last (CLS/EOS)
    mean_no_special = hidden_states[:, 1:-1, :].mean(dim=1).cpu()

    # Model output should match mean_no_special, not mean_all
    output = model.embed_sequence(text).cpu()

    assert torch.allclose(output, mean_no_special, atol=1e-5), \
        "Output should exclude CLS/EOS tokens"
    assert not torch.allclose(output, mean_all, atol=1e-5), \
        "Output should differ from mean including special tokens"


@torch.no_grad()
def test_rinalmo_single_nucleotide(rinalmo):
    """Test embedding a single nucleotide."""
    model, embed_dim = rinalmo
    output = model.embed_sequence("A").cpu()
    assert output.shape == (1, embed_dim)
    assert not torch.isnan(output).any()


@torch.no_grad()
def test_rinalmo_max_length_boundary(rinalmo):
    """Test sequence exactly at max_length boundary."""
    model, embed_dim = rinalmo
    effective_max = model.max_length - 2

    # Exactly at boundary (1 chunk)
    seq_at_boundary = "ACUG" * (effective_max // 4)
    output1 = model.embed_sequence(seq_at_boundary).cpu()
    assert output1.shape == (1, embed_dim)

    # One over boundary (2 chunks)
    seq_over_boundary = seq_at_boundary + "A"
    output2 = model.embed_sequence(seq_over_boundary).cpu()
    assert output2.shape == (1, embed_dim)

    # Outputs should be different
    assert not torch.allclose(output1, output2, atol=1e-5)


@torch.no_grad()
def test_rinalmo_embed_ragged_agg(rinalmo_giga):
    """Test embed with identity agg_fn returns per-token embeddings (ragged)."""
    seqs = ["ATGATG", "GCGCGCGCGCGC"]
    out = rinalmo_giga.embed(seqs, agg_fn=lambda x, **kwargs: x)
    assert out[0].dim() == 2  # (num_tokens, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different token counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_rinalmo_gradient_flow(rinalmo_giga):
    """Test that gradients can flow through the model."""
    rinalmo_giga.set_train_mode()

    out = rinalmo_giga.embed(["ATGATG"])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in rinalmo_giga.model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    rinalmo_giga.set_inference_mode()


def test_rinalmo_extract_structure(rinalmo_giga):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    h, s = rinalmo_giga.extract(["ATGATG"], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_rinalmo_extract_layer_selection(rinalmo_giga):
    """Requesting layers=[0] returns exactly 1 layer."""
    h, _ = rinalmo_giga.extract(["ATGATG"], layers=[0])
    assert len(h) == 1


def test_rinalmo_extract_attention_weights(rinalmo_giga):
    """return_attentions=True yields (H, T, T) tensors with rows summing to 1."""
    h, s = rinalmo_giga.extract(["ATGATG"], layers=[0], return_attentions=True)
    layer = next(iter(s))
    attn = s[layer]
    assert attn is not None
    w = attn[0][0]
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
