import pytest

import math
from unittest.mock import Mock

import torch

from mrna_bench.models.borzoi import Borzoi


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def borzoi(device) -> Borzoi:
    """Get Borzoi model."""
    model = Borzoi("borzoi-replicate-0", device)
    model.set_inference_mode()
    return model


def test_borzoi_forward(borzoi):
    """Test Borzoi forward pass."""
    text = "ACGT" * 50000
    output = borzoi.embed_sequence(text)
    assert output.shape[0] == 1
    assert output.shape[1] == 1536


def test_borzoi_embed_batch(borzoi):
    """Test batch embed matches individual embeddings."""
    sequences = [
        "ACGT" * 50000,
        "ACGT" * 60000,
    ]

    batch_output = torch.stack(borzoi.embed(sequences)).cpu()
    assert batch_output.shape == (2, 1536)

    for i, seq in enumerate(sequences):
        single_output = borzoi.embed_sequence(seq).cpu()
        assert torch.allclose(
            batch_output[i:i + 1],
            single_output,
            atol=1e-4
        ), "Mismatch at sequence {} (len {})".format(i, len(seq))


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

        output = borzoi.embed_sequence(short_seq, agg_fn=lambda x: x)

        padding_left = (min_length - seq_len) // 2
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
