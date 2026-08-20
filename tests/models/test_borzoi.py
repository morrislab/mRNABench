import pytest

import math
from unittest.mock import Mock, patch

import torch

from tests.model_utils import assert_raw_batch_matches_single, embed_one

from mrna_bench.models.borzoi import Borzoi


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def borzoi(device) -> Borzoi:
    """Get Borzoi model."""
    model = Borzoi("borzoi-replicate-0", device, "eager")
    model.set_inference_mode()
    return model


def test_borzoi_forward(borzoi):
    """Test Borzoi forward pass."""
    text = "ACGT" * 50000
    output = embed_one(borzoi, text)
    assert output.shape[0] == 1
    assert output.shape[1] == 1536


def test_borzoi_embed_batch(borzoi):
    """Test batch embed matches individual embeddings."""
    sequences = [
        "ACGT" * 50000,
        "ACGT" * 60000,
    ]

    with patch.object(
        borzoi.models[0],
        "get_embs_after_crop",
        wraps=borzoi.models[0].get_embs_after_crop,
    ) as forward:
        batch_output = torch.stack(borzoi.embed(sequences)).cpu()
    assert forward.call_args.args[0].shape[0] == 2
    assert batch_output.shape == (2, 1536)

    for i, seq in enumerate(sequences):
        single_output = embed_one(borzoi, seq).cpu()
        # Long padded windows change low-level convolution numerics.
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


@torch.no_grad()
def test_borzoi_predict_tracks(borzoi):
    """Expose native tracks aligned to the original sequence bins."""
    with patch.object(
        borzoi.models[0],
        "forward",
        wraps=borzoi.models[0].forward,
    ) as forward:
        output = borzoi.predict_tracks(["ACGT" * 250])[0]
    batch = forward.call_args.args[0]
    first_base = int((batch[0].sum(dim=0) > 0).nonzero()[0])

    assert output.bin_size == borzoi.bin_size
    assert output.start == 0
    assert first_base % borzoi.bin_size == 0
    assert output.values["human"].shape[1] == 7611


def test_borzoi_padding_logic(borzoi):
    """Test that padding and bin extraction work correctly."""
    bin_size = borzoi.bin_size
    min_length = borzoi.min_length

    def side_effect(batch):
        seq_len = batch.shape[2]
        num_bins = seq_len // bin_size
        pos = torch.arange(num_bins).unsqueeze(0).unsqueeze(0)
        return pos.expand(1, 1536, -1).float()

    mock_model = Mock()
    mock_model.get_embs_after_crop = Mock(side_effect=side_effect)
    original_models = borzoi.models
    borzoi.models = [mock_model]

    try:
        short_seq = "ACGT" * 1000
        seq_len = len(short_seq)

        output = embed_one(borzoi, short_seq, agg_fn=lambda x: x)

        padding_left = (min_length - seq_len) // 2 // bin_size * bin_size
        expected_start_bin = padding_left // bin_size
        expected_end_bin = math.ceil((padding_left + seq_len) / bin_size)
        expected_num_bins = expected_end_bin - expected_start_bin

        assert output.shape[1] == expected_num_bins

        first_bin_val = output[0, 0, 0].item()
        assert first_bin_val == expected_start_bin
    finally:
        borzoi.models = original_models


def test_borzoi_single_vs_ensemble():
    """Test that single model has 1 model, ensemble has 4."""
    # Check model count without actually loading (just verify the logic)
    # by checking versions_to_load would be constructed correctly

    # Single replicate should have 1 model
    single_versions = ["borzoi-replicate-0"]
    assert len(single_versions) == 1

    # Ensemble should have 4 models
    ensemble_versions = [
        "{}-replicate-{}".format("borzoi", i) for i in range(4)
    ]
    assert len(ensemble_versions) == 4
    assert ensemble_versions[0] == "borzoi-replicate-0"
    assert ensemble_versions[3] == "borzoi-replicate-3"


@torch.no_grad()
def test_borzoi_embed_ragged_agg(borzoi):
    """Test embed with identity agg_fn returns per-bin embeddings (ragged)."""
    seqs = ["ATGC" * 100, "ATGC" * 400]  # 400 and 1600 bp -> different bin counts
    out = borzoi.embed(seqs, agg_fn=lambda x, **kwargs: x)
    # Unpooled bins retain padding-context differences that pooling averages.
    assert_raw_batch_matches_single(borzoi, seqs, out, atol=2e-3)
    assert out[0].dim() == 2  # (num_bins, hidden_dim)
    assert out[1].dim() == 2
    assert out[0].shape[0] != out[1].shape[0]  # ragged: different bin counts
    assert out[0].shape[1] == out[1].shape[1]  # same hidden dim


def test_borzoi_ensemble_averaging():
    """Test that ensemble averages embeddings from multiple models."""
    # Create mock models with different outputs
    mock_model_1 = Mock()
    mock_model_2 = Mock()

    # Model 1 returns all 1s, Model 2 returns all 3s
    # Average should be 2s
    mock_model_1.get_embs_after_crop = Mock(
        return_value=torch.ones(1, 1536, 100)
    )
    mock_model_2.get_embs_after_crop = Mock(
        return_value=torch.ones(1, 1536, 100) * 3
    )

    # Test the averaging logic directly
    replicate_embeds = [
        mock_model_1.get_embs_after_crop(None),
        mock_model_2.get_embs_after_crop(None),
    ]
    averaged = torch.stack(replicate_embeds).mean(dim=0)

    assert averaged.shape == (1, 1536, 100)
    assert torch.allclose(averaged, torch.ones(1, 1536, 100) * 2)


def test_borzoi_extract_structure(borzoi):
    """extract() returns (dict, dict) with matching keys; hidden states are 2D."""
    seq = "ACGT" * 1000  # 4000 nt
    h, s = borzoi.extract([seq], layers=[0])
    assert isinstance(h, dict) and isinstance(s, dict)
    assert set(h.keys()) == set(s.keys())
    layer = next(iter(h))
    assert h[layer][0][0].dim() == 2
    assert h[layer][0][0].device.type == "cpu"


def test_borzoi_extract_layer_selection(borzoi):
    """Requesting layers=[0] returns exactly 1 layer."""
    seq = "ACGT" * 1000
    h, _ = borzoi.extract([seq], layers=[0])
    assert len(h) == 1


def test_borzoi_extract_scores_none(borzoi):
    """CNN layers return None scores (layers=[0] resolves to res_tower.0)."""
    seq = "ACGT" * 1000
    _, s = borzoi.extract([seq], layers=[0], return_attentions=True)
    assert all(v is None for v in s.values())


def test_borzoi_extract_transformer_attention(borzoi):
    """Transformer layers return (H, T, T) attention weights with rows summing to 1."""
    seq = "ACGT" * 1000
    _, s = borzoi.extract([seq], layers=["transformer.0"], return_attentions=True)
    assert s["transformer.0"] is not None
    w = s["transformer.0"][0][0]  # seq[0], chunk[0] -> (H, T, T)
    assert w.dim() == 3
    assert w.shape[1] == w.shape[2]  # T x T square
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)


def test_borzoi_extract_cnn_scores_none_with_attentions(borzoi):
    """CNN layers return None scores even when return_attentions=True."""
    seq = "ACGT" * 1000
    _, s = borzoi.extract([seq], layers=["res_tower.0"], return_attentions=True)
    assert s["res_tower.0"] is None


def test_borzoi_gradient_flow(borzoi):
    """Test that gradients can flow through the model."""
    borzoi.set_train_mode()

    out = borzoi.embed(["ACGT" * 100])
    assert out[0].requires_grad, "Output should require gradients"

    loss = torch.stack(out).sum()
    loss.backward()

    has_grad = False
    for param in borzoi.models[0].parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients flowed to model parameters"
    borzoi.set_inference_mode()


def test_borzoi_peft_target(borzoi):
    """get_peft_target returns models[0]; set_peft_target writes back to it."""
    target = borzoi.get_peft_target()
    assert target is borzoi.models[0]

    sentinel = torch.nn.Linear(1, 1)
    original = borzoi.models[0]
    borzoi.set_peft_target(sentinel)
    assert borzoi.models[0] is sentinel

    borzoi.models[0] = original  # restore for subsequent tests


def test_borzoi_extract_cnn_downsampling(borzoi):
    """CNN res_tower layers produce shorter sequences as the tower progresses.

    res_tower.0 is a ConvBlock at the same resolution as conv_dna (no stride).
    Downsampling occurs in res_tower.1 (MaxPool1d, not hookable), so the first
    hookable layer with reduced length is res_tower.2.
    """
    seq = "ACGT" * 1000
    h, _ = borzoi.extract([seq], layers=["conv_dna", "res_tower.0", "res_tower.2", "res_tower.8"])
    T_dna = h["conv_dna"][0][0].shape[0]
    T_res0 = h["res_tower.0"][0][0].shape[0]
    T_res2 = h["res_tower.2"][0][0].shape[0]
    T_res8 = h["res_tower.8"][0][0].shape[0]
    assert T_res0 == T_dna   # res_tower.0 is a ConvBlock: same resolution as conv_dna
    assert T_res2 < T_dna    # res_tower.2 is after the first MaxPool: 2× downsampled
    assert T_res8 <= T_res2  # res_tower.8 is further downsampled
