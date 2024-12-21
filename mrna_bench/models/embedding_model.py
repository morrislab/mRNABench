from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch


class EmbeddingModel(ABC):
    def __init__(self, model_version: str, device: torch.device):
        self.model_version = model_version
        self.device = device

        self.is_sixtrack = False

        print("Disabling autograd for inference.")
        torch.autograd.set_grad_enabled(False)

    @abstractmethod
    def embed_sequence(
        self,
        sequence: str,
        overlap: int,
        agg_fn: Callable,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_model_short_name(self):
        pass

    def chunk_sequence(
        self,
        sequence: str,
        chunk_length: int,
        overlap_size: int
    ) -> list[str]:
        """Split sequence into chunks of specified length with given overlap.

        Args:
            sequence: The input string sequence to be chunked.
            chunk_length: The length of each chunk.
            overlap_size: The number of overlapping characters between chunks.

        Returns:
            A list of string chunks, where each chunk has the specified length.
        """
        step_size = chunk_length - overlap_size

        chunks = []
        for i in range(0, len(sequence), step_size):
            chunk = sequence[i:i + chunk_length]
            chunks.append(chunk)

        # Ensure the last incomplete chunk is included (if not already added)
        if len(sequence) > len(chunks) * step_size:
            chunks.append(sequence[-chunk_length:])

        return chunks
