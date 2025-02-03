from collections import namedtuple

import pytest
from unittest.mock import patch

import numpy as np
import torch

from mrna_bench.models.utrbert import UTRBERT


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Get torch cuda device if available, else use cpu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@pytest.fixture(scope="module")
def utrbert3mer(device) -> UTRBERT:
    """Get UTR-BERT model with 3mer tokenization."""
    return UTRBERT("utrbert-3mer", device)


@pytest.fixture(scope="module")
def utrbert6mer(device) -> UTRBERT:
    """Get UTR-BERT model with 6mer tokenization."""
    return UTRBERT("utrbert-6mer", device)


@pytest.fixture(scope="module")
def utrbert_utronly(device) -> UTRBERT:
    """Get UTR-BERT model."""
    return UTRBERT("utrbert-3mer-utronly", device)


def test_utrbert_forward(utrbert3mer):
    """Test 3'UTR-BERT forward pass."""
    assert utrbert3mer.is_sixtrack is False

    input_seq = "ATGATG"

    with patch.object(
        utrbert3mer,
        "tokenizer",
        wraps=utrbert3mer.tokenizer
    ) as mock:
        output = utrbert3mer.embed_sequence(input_seq, overlap=0)
        mock.assert_called_once_with("AUGAUG", return_tensors="pt")

        assert output.shape == (1, 768)


def test_utrbert_get_utr(utrbert_utronly):
    """Test helper method from 3'UTR-BERT to retrieve 3'UTR region."""
    input_seq = "ATGATGATG"

    cds_1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])

    utr1 = utrbert_utronly.get_threeprime_utr(input_seq, cds_1)
    assert utr1 == "TGATG"

    cds_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    utr2 = utrbert_utronly.get_threeprime_utr(input_seq, cds_2)
    assert utr2 == input_seq

    # If UTR is shorter than kmers, return whole sequence
    cds_3 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    utr3 = utrbert_utronly.get_threeprime_utr(input_seq, cds_3)
    assert utr3 == input_seq


def test_utrbert_utronly_forward(utrbert_utronly):
    """Test 3'UTR-BERT utr only forward pass."""
    assert utrbert_utronly.is_sixtrack is True

    input_seq = "ATGATG"

    with patch.object(
        utrbert_utronly,
        "tokenizer",
        wraps=utrbert_utronly.tokenizer
    ) as mock:
        output = utrbert_utronly.embed_sequence_sixtrack(
            input_seq,
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0] * len(input_seq)),
            overlap=0
        )
        mock.assert_called_once_with("AUG", return_tensors="pt")

        assert output.shape == (1, 768)


def test_utrbert_forward_chunked_3mer(utrbert3mer):
    """Test 3'UTR-BERT forward pass with chunked sequence."""
    spillover = 100
    input_seq = "A" * (utrbert3mer.max_length + spillover)

    token_len = len(input_seq) - (utrbert3mer.kmer_size - 1)

    # Check that all only first token (CLS) is used for aggregation
    ground_truth = torch.zeros(1, 768)

    def side_effect(input_ids, attention_mask):
        if mock.call_count == 1:
            assert input_ids.shape[1] == utrbert3mer.max_length
        elif mock.call_count == 2:
            assert input_ids.shape[1] == token_len - utrbert3mer.max_length + 4
        else:
            raise AssertionError("Unexpected call count")

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 768)

        mock_out = {"hidden_states": [pos]}

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        mock_out = MockOut(last_hidden_state=pos)
        return mock_out

    with patch.object(utrbert3mer, "model", side_effect=side_effect) as mock:
        output = utrbert3mer.embed_sequence(input_seq, overlap=0)

        assert torch.allclose(output, ground_truth)


def test_utrbert_forward_chunked_6mer(utrbert6mer):
    """Test 3'UTR-BERT forward pass with chunked sequence."""
    spillover = 100
    input_seq = "A" * (utrbert6mer.max_length + spillover)

    token_len = len(input_seq) - (utrbert6mer.kmer_size - 1)

    # Check that all only first token (CLS) is used for aggregation
    ground_truth = torch.zeros(1, 768)

    def side_effect(input_ids, attention_mask):
        if mock.call_count == 1:
            assert input_ids.shape[1] == utrbert6mer.max_length
        elif mock.call_count == 2:
            assert input_ids.shape[1] == token_len - utrbert6mer.max_length + 4
        else:
            raise AssertionError("Unexpected call count")

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 768)

        mock_out = {"hidden_states": [pos]}

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        mock_out = MockOut(last_hidden_state=pos)
        return mock_out

    with patch.object(utrbert6mer, "model", side_effect=side_effect) as mock:
        output = utrbert6mer.embed_sequence(input_seq, overlap=0)

        assert torch.allclose(output, ground_truth)


def test_utrbert_forward_chunked_3mer_overlap(utrbert3mer):
    """Test 3'UTR-BERT forward pass with chunked sequence and overlap."""
    spillover = 100
    overlap = 25
    max_len = utrbert3mer.max_length

    input_seq = "A" * (max_len + spillover)

    token_len = len(input_seq) - (utrbert3mer.kmer_size - 1)

    # Check that all only first token (CLS) is used for aggregation
    ground_truth = torch.zeros(1, 768)

    def side_effect(input_ids, attention_mask):
        if mock.call_count == 1:
            assert input_ids.shape[1] == max_len
        elif mock.call_count == 2:
            assert input_ids.shape[1] == token_len - max_len + 4 + overlap
        else:
            raise AssertionError("Unexpected call count")

        pos = torch.arange(input_ids.shape[1]).unsqueeze(0).unsqueeze(-1)
        pos = pos.float().repeat(1, 1, 768)

        mock_out = {"hidden_states": [pos]}

        MockOut = namedtuple("MockOut", ["last_hidden_state"])
        mock_out = MockOut(last_hidden_state=pos)
        return mock_out

    with patch.object(utrbert3mer, "model", side_effect=side_effect) as mock:
        output = utrbert3mer.embed_sequence(input_seq, overlap=overlap)

        assert torch.allclose(output, ground_truth)
