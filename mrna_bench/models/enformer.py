from collections.abc import Callable
from functools import partial
import math

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.datasets.dataset_utils import str_to_ohe


class Enformer(EmbeddingModel):
    """Inference wrapper for Enformer.

    Enformer is a deep learning model for predicting functional genomic
    activity from DNA sequence. It uses a transformer-based architecture
    built on convolutional layers and self-attention mechanisms to model
    long-range interactions in the genome. Enformer is trained on 200 kb
    genomic windows and predicts a range of functional readouts across
    multiple human and mouse datasets. Here we use the PyTorch
    implementation of Enformer from EleutherAI, which is based on the
    original Enformer model from DeepMind.

    Link: https://github.com/google-deepmind/deepmind-research
        (Under enformer directory)
    Link: https://github.com/lucidrains/enformer-pytorch
    """

    default_version = "enformer-official-rough"
    valid_versions = ["enformer-official-rough"]

    lora_target_modules = ["to_q", "to_k", "to_v", "to_out"]

    prediction_window = 114_688  # embedding is of the center 114688 bases
    max_length = 196_608  # can take sequences up to 196608 bases
    bin_size = 128  # embedding is in 128 base bins

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Enformer.

        Args:
            model_version: Version of model used. Valid versions: {
                "enformer-official-rough"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from enformer_pytorch import from_pretrained
        except ImportError:
            raise ImportError("Enformer missing required dependencies.")

        self.model = from_pretrained(
            f'EleutherAI/{model_version}',
            cache_dir=get_model_weights_path()
        ).to(device)

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Embed sequence using Enformer, excluding padded regions.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Tensor representing embedded sequence.
        """
        _, _ = cds, splice

        def center_padding(seq: str, length: int) -> tuple[str, int]:
            """Center pad sequence to a given length."""
            padding_left = (length - len(seq)) // 2
            padding_right = length - len(seq) - padding_left

            return "N" * padding_left + seq + "N" * padding_right, padding_left

        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        for chunk in chunks:
            padded_chunk, padding_left = center_padding(chunk, self.max_length)

            # first OHE sequence chunk
            batch = torch.tensor(
                str_to_ohe(padded_chunk),
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            _, embedded_chunk = self.model(
                batch,
                return_embeddings=True,
                target_length=-1
            )

            # extract embedding portion corresponding to original unpadded seq
            start_bin = padding_left // self.bin_size
            end_bin = math.ceil((padding_left + len(chunk)) / self.bin_size)

            embedding = embedded_chunk[:, start_bin:end_bin, :]

            embedding_chunks.append(embedding)

        embedding = torch.cat(embedding_chunks, dim=1).squeeze(0)

        aggregate_embedding = agg_fn(embedding).unsqueeze(0)
        return aggregate_embedding

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using Enformer.

        Processes sequences one at a time due to memory constraints
        at 196k bp sequence lengths.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (3072,)

        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding.squeeze(0))

        return all_embeddings
