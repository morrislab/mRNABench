from collections.abc import Callable
from functools import partial

import math

import numpy as np

import torch
import torch.nn as nn

from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.datasets.dataset_utils import str_to_ohe


class MixerModel(nn.Module):
    """Implementation of Mamber Mixer condensed from Mamba repo."""

    def __init__(self, d_model: int, n_layer: int, input_dim: int):
        """Initialize Mixer model.

        Args:
            d_model: Dimension of model.
            n_layer: Number of layers.
            input_dim: Input dimension.
        """
        super().__init__()

        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            from mamba_ssm.modules.block import Block
        except ImportError:
            try:
                from mamba_ssm.modules.mamba_simple import Block, Mamba
            except ImportError:
                raise ImportError(
                    "Install base_models optional dependency to use NaiveMamba."
                )

        # Feature detection for mamba-ssm version compatibility
        has_mlp = "mlp_cls" in Block.__init__.__code__.co_varnames

        self.embedding = nn.Linear(input_dim, d_model)

        blocks = []
        for i in range(n_layer):
            mix_cls = partial(Mamba, layer_idx=i)
            if has_mlp:
                # mamba-ssm >= 2.0: requires mlp_cls
                block = Block(d_model, mix_cls, mlp_cls=nn.Identity)
            else:
                # mamba-ssm < 2.0: only mixer_cls
                block = Block(d_model, mix_cls)
            block.layer_idx = i
            blocks.append(block)
        self.layers = nn.ModuleList(blocks)

        self.norm_f = nn.LayerNorm(d_model)

        self.apply(partial(self._init_weights, n_layer=n_layer))

    def forward(self, x: torch.Tensor, channel_last=False) -> torch.Tensor:
        """Mamba mixer forward pass.

        Args:
            x: Input tensor.
            channel_last: Whether the input tensor is in channel last format.

        Returns:
            Output tensor.
        """
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        res = None
        for layer in self.layers:
            hidden_states, res = layer(hidden_states, res)

        res = (hidden_states + res) if res is not None else hidden_states
        hidden_states = self.norm_f(res.to(dtype=self.norm_f.weight.dtype))

        hidden_states = hidden_states

        return hidden_states

    @staticmethod
    def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,
    ):
        """Initialize weights of Mamba model."""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class NaiveMamba(EmbeddingModel):
    """Naive Mamba which uses Mamba random initialization without training."""

    default_version = "naive-mamba"
    valid_versions = ["naive-mamba"]

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NaiveMamba model.

        Args:
            model_version: Unused.
            device: PyTorch device to send model to.
        """
        _ = model_version
        super().__init__("naive-mamba", device)

        torch.random.manual_seed(0)
        np.random.seed(0)
        self.model = MixerModel(
            d_model=64,
            n_layer=3,
            input_dim=6,
        ).to(device)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using NaiveMamba.

        NaiveMamba requires 6-track input (sequence + CDS + splice).

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks for sequences (required).
            splice: List of splice site tracks for sequences (required).
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (1, 64)
        """
        if cds is None or splice is None:
            raise ValueError("NaiveMamba requires cds and splice tracks.")

        batch_inputs = []
        lengths = []
        for seq, c, s in zip(sequences, cds, splice):
            ohe_sequence = str_to_ohe(seq)
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
        ).transpose(1, 2)

        hidden_states = self.model(batch_tensor)

        embeddings = []
        for i, length in enumerate(lengths):
            seq_hidden = hidden_states[i, :length, :]
            embeddings.append(agg_fn(seq_hidden))

        return embeddings
