import pytest
from unittest.mock import patch

import numpy as np

pytest.importorskip("torch")
import torch
from mrna_bench.models.rnafm import RNAFM


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def rnamodel(device) -> RNAFM:
    """Get RNA-FM model."""
    return RNAFM("rna-fm", device)


@pytest.fixture(scope="module")
def mrnamodel(device) -> RNAFM:
    """Get mRNA-FM model."""
    return RNAFM("mrna-fm", device)


def test_rnafm_forward(rnamodel):
    """Test RNA-FM forward pass."""
    assert rnamodel.max_length == 1024
    assert rnamodel.is_sixtrack is False

    out = rnamodel.embed_sequence("ATGATG")
    assert out.shape == (1, 640)


def test_mrnafm_forward(mrnamodel):
    """Test mRNA-FM  initialization and forward pass."""
    assert mrnamodel.max_length == 1024
    assert mrnamodel.is_sixtrack is True

    out = mrnamodel.embed_sequence_sixtrack(
        "ATGATG",
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 1, 0, 0])
    )
    assert out.shape == (1, 1280)


def test_rnafm_forward_replace(rnamodel):
    """Test RNA-FM forward pass converts T->U."""
    with patch.object(
        rnamodel,
        "chunk_sequence",
        side_effect=rnamodel.chunk_sequence
    ) as mock:
        rnamodel.embed_sequence("ATGATG")
        mock.assert_called_once_with("AUGAUG", rnamodel.max_length - 2)


def test_mrnafm_forward_replace(mrnamodel):
    """Test mRNA-FM forward pass converts T->U."""
    with patch.object(
        mrnamodel,
        "chunk_sequence",
        side_effect=mrnamodel.chunk_sequence
    ) as mock:
        mrnamodel.embed_sequence_sixtrack(
            "ATGATG",
            np.array([1, 0, 0] * 2),
            np.array([0] * 6),
        )
        mock.assert_called_once_with("AUGAUG", 3 * (mrnamodel.max_length - 2))


def test_rnafm_forward_chunking(rnamodel):
    """Test RNA-FM forward pass with chunking."""
    spillover = 10
    input_seq = "A" * (2 * rnamodel.max_length + spillover)

    def side_effect(tokens, repr_layers=[12]):
        if mock_model.call_count == 1:
            assert len(tokens[0]) == rnamodel.max_length - 1
            assert tokens[0][0] == 0
            assert tokens[0][-1] != 2
        elif mock_model.call_count == 2:
            assert len(tokens[0]) == rnamodel.max_length - 2
            assert tokens[0][0] != 0
            assert tokens[0][-1] != 2
        else:
            # 2 tokens per previous chunk, plus extra token at end
            assert len(tokens[0]) == spillover + 5
            assert tokens[0][0] != 0
            assert tokens[0][-1] == 2
        return {"representations": [torch.zeros(1, 640)] * 13}

    with patch.object(rnamodel, "model") as mock_model:
        mock_model.side_effect = side_effect
        rnamodel.embed_sequence(input_seq)


def test_mrna_forward_cds_slice(mrnamodel):
    """Test mRNA-FM forward pass with CDS slice."""
    input_seq = "A" * 30 + "T" * 30 + "G" * 40
    cds = np.array([0] * 30 + [1, 0, 0] * 10 + [0] * 40)
    splice = np.array([0] * 100)

    def side_effect(tokens, repr_layers=[12]):
        assert torch.all(tokens[0][1:-1] == 23)
        return {"representations": [torch.zeros(1, 1280)] * 13}

    with patch.object(mrnamodel, "model") as mock_model:
        mock_model.side_effect = side_effect
        mrnamodel.embed_sequence_sixtrack(input_seq, cds, splice)


def test_mrnafm_forward_chunking(mrnamodel):
    """Test mRNA-FM forward pass with chunking."""
    spillover = 10
    input_seq = "T" * (mrnamodel.max_length * 3 + spillover * 3)
    cds = np.array([1, 0, 0] * (mrnamodel.max_length + spillover))
    splice = np.array([0] * 3 * (mrnamodel.max_length + spillover))

    def side_effect(tokens, repr_layers=[12]):
        if mock_model.call_count == 1:
            assert len(tokens[0]) == mrnamodel.max_length - 1
            assert tokens[0][0] == 0
            assert tokens[0][-1] != 2
        elif mock_model.call_count == 2:
            assert len(tokens[0]) == spillover + 3
            assert tokens[0][0] != 0
            assert tokens[0][-1] == 2

        return {"representations": [torch.zeros(1, 1280)] * 13}

    with patch.object(mrnamodel, "model") as mock_model:
        mock_model.side_effect = side_effect
        mrnamodel.embed_sequence_sixtrack(input_seq, cds, splice)


def test_get_cds_full(mrnamodel):
    """Test get_cds method."""
    sequence = "CCGATGCCG"
    cds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])

    cds_seq = mrnamodel.get_cds(sequence, cds)
    assert cds_seq == "ATG"


def test_get_cds_missing(mrnamodel):
    """Test get_cds method when cds is missing."""
    sequence_1 = "CCGATGCCG"
    cds_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    sequence_2 = "CCGATGCC"
    cds_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    cds_seq_1 = mrnamodel.get_cds(sequence_1, cds_1)
    assert cds_seq_1 == "CCGATGCCG"

    cds_seq_2 = mrnamodel.get_cds(sequence_2, cds_2)
    assert cds_seq_2 == "CCGATG"


def test_get_cds_irregular(mrnamodel):
    """Test get_cds method when cds is not a multiple of 3."""
    sequence = "CCGATGCCG"
    cds_1 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0])
    cds_2 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1])

    cds_1_seq = mrnamodel.get_cds(sequence, cds_1)
    cds_2_seq = mrnamodel.get_cds(sequence, cds_2)

    assert cds_1_seq == "TGC"
    assert cds_2_seq == "GCC"
