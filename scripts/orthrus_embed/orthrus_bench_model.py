import os
import torch
import numpy as np
from mrna_bench.models import EmbeddingModel
from collections.abc import Callable
from mrna_bench.datasets.dataset_utils import str_to_ohe
from functools import partial
from orthrus_src import load_model

class Orthrus(EmbeddingModel):

    valid_attn_implementations = None
    default_attn_implementation = None
    hookable_layer_patterns = [r"layers\.\d+"]

    lora_target_modules = ["in_proj", "out_proj", "x_proj", "dt_proj"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-").replace("best-", "")

    def __init__(self,
        model_version: str,
        checkpoint : str,
        device: torch.device,
        model_repository: str,
        attn_implementation: str | None,
    ):

        self.default_version = model_version + "_" + checkpoint.replace(".ckpt", "")
        self.valid_versions = [model_version + "_" + checkpoint.replace(".ckpt", "")]

        super().__init__(
                model_version + "_" + checkpoint.replace(".ckpt", ""),
                device
            )

        model = load_model(
            f"{os.path.join(model_repository, '')}{model_version}",
            checkpoint_name=checkpoint,
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
        if "6-track" in self.get_model_short_name(self.model_version):
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
            - default (mean): (512,)
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
            - default (mean): (512,)
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
