from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class Orthrus(EmbeddingModel):
    """Inference wrapper for Orthrus.

    Orthrus is a RNA foundation model trained using a Mamba backbone. It uses
    a contrastive learning pre-training objective that maximizes similarity
    between RNA splice isoforms and orthologous transcripts. Input length is
    unconstrained due to use of Mamba.

    Link: https://github.com/bowang-lab/Orthrus
    """

    default_version = "orthrus-large-6-track"
    valid_versions = ["orthrus-large-6-track", "orthrus-large-4-track"]

    lora_target_modules = ["in_proj", "out_proj", "x_proj", "dt_proj"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Orthrus model.

        Args:
            model_version: Version of Orthrus to load. Valid values are: {
                "orthrus-large-4-track",
                "orthrus-large-6-track"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use Orthrus."
            )

        model_hf_path = "quietflamingo/{}".format(model_version)
        model = AutoModel.from_pretrained(
            model_hf_path,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = model.to(device)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using Orthrus.

        Routes to 4-track or 6-track embedding based on model version.
        - orthrus-base-4-track: Uses sequence only (ignores cds/splice)
        - orthrus-large-6-track: Requires cds and splice tracks

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks (required for 6-track model).
            splice: List of splice site tracks (required for 6-track model).
            agg_fn: Unused (Orthrus uses internal pooling).

        Returns:
            Orthrus embeddings with shape (batch_size, hidden_dim).
        """
        _ = agg_fn

        if self.model_version == "orthrus-large-6-track":
            if cds is None or splice is None:
                raise ValueError(
                    "Orthrus 6-track model requires cds and splice tracks."
                )
            return self._embed_sixtrack(sequences, cds, splice)
        else:
            return self._embed_fourtrack(sequences)

    def _embed_fourtrack(self, sequences: list[str]) -> torch.Tensor:
        """Embed sequences using 4-track Orthrus.

        Args:
            sequences: List of sequences to embed.

        Returns:
            Orthrus embeddings with shape (batch_size, hidden_dim).
        """
        batch_inputs = []
        lengths = []

        for seq in sequences:
            ohe_sequence = self.model.seq_to_oh(seq)
            batch_inputs.append(ohe_sequence)
            lengths.append(len(seq))

        max_len = max(lengths)
        padded_inputs = []
        for inp in batch_inputs:
            if inp.shape[0] < max_len:
                padding = torch.zeros((max_len - inp.shape[0], 4))
                inp = torch.vstack((inp, padding))
            padded_inputs.append(inp)

        batch_tensor = torch.stack(padded_inputs, dim=0).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32).to(self.device)

        embeddings = self.model.representation(
            batch_tensor,
            lengths_tensor,
            channel_last=True
        )

        return embeddings

    def _embed_sixtrack(
        self,
        sequences: list[str],
        cds: list[np.ndarray],
        splice: list[np.ndarray],
    ) -> torch.Tensor:
        """Embed sequences using 6-track Orthrus.

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks for sequences.
            splice: List of splice site tracks for sequences.

        Returns:
            Orthrus embeddings with shape (batch_size, hidden_dim).
        """
        batch_inputs = []
        lengths = []

        for seq, c, s in zip(sequences, cds, splice):
            ohe_sequence = self.model.seq_to_oh(seq).numpy()
            model_input = np.hstack((
                ohe_sequence,
                c.reshape(-1, 1),
                s.reshape(-1, 1)
            ))
            batch_inputs.append(model_input)
            lengths.append(len(seq))

        max_len = max(lengths)
        padded_inputs = []
        for inp in batch_inputs:
            if inp.shape[0] < max_len:
                padding = np.zeros((max_len - inp.shape[0], 6))
                inp = np.vstack((inp, padding))
            padded_inputs.append(inp)

        batch_tensor = torch.tensor(
            np.stack(padded_inputs),
            dtype=torch.float32,
            device=self.device
        )
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32).to(self.device)

        embeddings = self.model.representation(
            batch_tensor,
            lengths_tensor,
            channel_last=True
        )

        return embeddings
