from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SupportsEmbedding(Protocol):
    """Protocol defining the interface for embedding models."""

    device: torch.device
    model: torch.nn.Module

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None,
        splice: list[np.ndarray] | None,
        agg_fn: Callable,
    ) -> list[torch.Tensor]:
        """Embed sequences, optionally using cds/splice tracks."""
        ...


class EmbeddingModel(SupportsEmbedding, ABC):
    """Wrapper class for embedding models used to represent sequences."""

    default_version: ClassVar[str]
    valid_versions: ClassVar[list[str]]
    model: torch.nn.Module

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Retrieve shortened name for model version.

        Override in subclass if the version name needs custom transformation.
        By default, replaces underscores with hyphens.

        Args:
            model_version: Version of model to fetch short name for.

        Returns:
            Shortened name of model version.
        """
        return model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize EmbeddingModel.

        Args:
            model_version: Version of embedding model to use.
            device: PyTorch device to send embedding model.

        Raises:
            ValueError: If model_version is not in valid_versions.
        """
        if model_version not in self.valid_versions:
            raise ValueError(
                "Invalid model version: {}. Valid versions: {}".format(
                    model_version, self.valid_versions
                )
            )
        self.model_version = model_version
        self.short_name = self.__class__.get_model_short_name(model_version)
        self.device = device

    def set_inference_mode(self):
        """Set model to inference mode with gradients disabled."""
        self.model.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        """Set model to training mode with gradients enabled."""
        self.model.train()
        torch.set_grad_enabled(True)

    @abstractmethod
    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> list[torch.Tensor]:
        """Embed sequences, optionally using cds/splice tracks.

        Args:
            sequences: List of nucleotide sequences to embed (uses DNA bases).
            cds: List of binary encodings of first nucleotide of each codon.
            splice: List of binary encodings of splice site locations.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embedded sequences with shape (batch_size x H).
        """
        pass

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> torch.Tensor:
        """Legacy wrapper for embed with a single sequence.

        Args:
            sequence: String of nucleotides to embed (uses DNA bases).
            cds: Binary encoding of first nucleotide of each codon in CDS.
            splice: Binary encoding of splice site locations.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embedded sequence with shape (1 x H).
        """
        cds_list = [cds] if cds is not None else None
        splice_list = [splice] if splice is not None else None
        embs = self.embed([sequence], cds_list, splice_list, agg_fn)
        result = torch.stack(embs)
        if result.dim() > 2:
            result = result.squeeze(0)
        return result

    def chunk_sequence(self, sequence: str, chunk_length: int) -> list[str]:
        """Split sequence into chunks of specified length with given overlap.

        Args:
            sequence: The input string sequence to be chunked.
            chunk_length: The length of each chunk.

        Returns:
            A list of string chunks, where each chunk has the specified length.
        """
        chunks = []
        for i in range(0, len(sequence), chunk_length):
            chunk = sequence[i:i + chunk_length]
            chunks.append(chunk)

        return chunks

    def chunk_tokens(
        self,
        sequence_tokens: list[int],
        chunk_length: int,
    ) -> list[list[int]]:
        """Chunk tokenized sequence into specified length.

        Args:
            sequence_tokens: The tokenized sequence to be chunked.
            chunk_length: The length of each chunk.

        Returns:
            A list of chunked tokens each with specified maximum length.
        """
        chunks = []
        for i in range(0, len(sequence_tokens), chunk_length):
            chunk = sequence_tokens[i:i + chunk_length]
            chunks.append(chunk)

        return chunks

    def _embed_with_chunking(
        self,
        sequences: list[str],
        max_chunk_length: int,
        embed_fn: Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]],
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> list[torch.Tensor]:
        """Embed sequences with chunking and reassembly.

        Handles the common pattern of chunking long sequences, running a
        single forward pass, and reassembling per-sequence embeddings.

        Args:
            sequences: List of sequences to embed.
            max_chunk_length: Maximum chunk length in nucleotides.
            embed_fn: Function that takes a list of sequence chunks and returns
                (hidden_states, pooling_mask) tensors. The pooling_mask
                indicates which tokens to include in aggregation
                (excludes padding and special tokens like CLS/SEP/EOS).
            agg_fn: Function to aggregate token embeddings (default: mean).

        Returns:
            Embeddings with shape (num_sequences, hidden_dim).
        """
        chunks = []
        chunk_counts = []

        for seq in sequences:
            seq_chunks = self.chunk_sequence(seq, max_chunk_length)
            chunks.extend(seq_chunks)
            chunk_counts.append(len(seq_chunks))

        hidden_states, pooling_mask = embed_fn(chunks)

        seq_embeddings = []
        chunk_ptr = 0

        for num_chunks in chunk_counts:
            seq_hidden = hidden_states[chunk_ptr:chunk_ptr + num_chunks]
            seq_mask = pooling_mask[chunk_ptr:chunk_ptr + num_chunks]

            hidden = seq_hidden.reshape(-1, seq_hidden.shape[-1])
            mask = seq_mask.reshape(-1).bool()

            masked_hidden = hidden[mask]
            seq_embeddings.append(agg_fn(masked_hidden))

            chunk_ptr += num_chunks

        return seq_embeddings
