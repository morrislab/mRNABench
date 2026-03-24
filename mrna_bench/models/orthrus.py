from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench.datasets.dataset_utils import str_to_ohe
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
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using Orthrus.

        Routes to 4-track or 6-track embedding based on model version.
        - orthrus-large-4-track: Uses sequence only (ignores cds/splice)
        - orthrus-large-6-track: Requires cds and splice tracks

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks (required for 6-track model).
            splice: List of splice site tracks (required for 6-track model).
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Orthrus embeddings with shape (batch_size, hidden_dim).
        """
        if self.model_version == "orthrus-large-6-track":
            if cds is None or splice is None:
                raise ValueError(
                    "Orthrus 6-track model requires cds and splice tracks."
                )
            return self._embed_sixtrack(sequences, cds, splice, agg_fn)
        else:
            return self._embed_fourtrack(sequences, agg_fn)

    def _embed_fourtrack(
        self,
        sequences: list[str],
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using 4-track Orthrus.

        Args:
            sequences: List of sequences to embed.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 512)
        """
        batch_inputs = []
        raw_lengths: list[int] = []

        for seq in sequences:
            ohe_sequence = torch.from_numpy(
                str_to_ohe(seq),
            ).float().to(self.device)
            batch_inputs.append(ohe_sequence)
            raw_lengths.append(len(seq))

        lengths = torch.tensor(
            raw_lengths,
            device=self.device,
            dtype=torch.float32
        )

        max_len = int(max(lengths))
        padded_inputs = []
        for inp in batch_inputs:
            if inp.shape[0] < max_len:
                padding = torch.zeros(
                    (max_len - inp.shape[0], 4),
                    device=self.device
                )
                inp = torch.vstack((inp, padding))
            padded_inputs.append(inp)

        batch_tensor = torch.stack(padded_inputs, dim=0).to(self.device)
        hidden_states = self.model.forward(batch_tensor, channel_last=True)

        pooling_mask = torch.arange(
            max_len,
            device=self.device
        ) < lengths[:, None]

        seq_embeddings = []
        for i in range(len(sequences)):
            seq_hidden = hidden_states[i][pooling_mask[i]]
            seq_embeddings.append(agg_fn(seq_hidden))

        return seq_embeddings

    def _embed_sixtrack(
        self,
        sequences: list[str],
        cds: list[np.ndarray],
        splice: list[np.ndarray],
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using 6-track Orthrus.

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks for sequences.
            splice: List of splice site tracks for sequences.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 512)
        """
        batch_inputs = []
        raw_lengths: list[int] = []

        for seq, c, s in zip(sequences, cds, splice):
            ohe_sequence = str_to_ohe(seq)
            model_input = np.hstack((
                ohe_sequence,
                c.reshape(-1, 1),
                s.reshape(-1, 1)
            ))
            batch_inputs.append(model_input)
            raw_lengths.append(len(seq))

        lengths = torch.tensor(
            raw_lengths,
            device=self.device,
            dtype=torch.float32
        )

        max_len = int(max(lengths))
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

        hidden_states = self.model.forward(batch_tensor, channel_last=True)

        pooling_mask = torch.arange(
            max_len,
            device=self.device
        ) < lengths[:, None]

        seq_embeddings = []
        for i in range(len(sequences)):
            seq_hidden = hidden_states[i][pooling_mask[i]]
            seq_embeddings.append(agg_fn(seq_hidden))

        return seq_embeddings
