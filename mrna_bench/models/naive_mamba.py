from typing import Callable
from functools import partial

import math

import numpy as np

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block


from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.datasets.dataset_utils import str_to_ohe


def create_block(d_model: int, layer_idx: int) -> Block:
    """Create Mamba block.

    Condensed implementation from Mamba github repo.

    Args:
        d_model: Dimension of model.
        layer_idx: Layer index.

    Returns:
        Mamba block.
    """
    mix_cls = partial(Mamba, layer_idx=layer_idx)
    block = Block(d_model, mix_cls)
    block.layer_idx = layer_idx
    return block


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

        self.embedding = nn.Linear(input_dim, d_model)

        blocks = [create_block(d_model, layer_idx=i) for i in range(n_layer)]
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

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NaiveMamba model.

        Args:
            model_version: Unused.
            device: PyTorch device to send model to.
        """
        _ = model_version
        super().__init__("naive-mamba", device)

        self.is_sixtrack = True

        torch.random.manual_seed(0)
        np.random.seed(0)
        self.model = MixerModel(
            d_model=64,
            n_layer=3,
            input_dim=6,
        ).to(device)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable | None = None
    ) -> torch.Tensor:
        """Embed sequence using four track Naive Mamba.

        Args:
            sequence: Sequence to embed.
            agg_fn: Currently unused.

        Returns:
            Naive Mamba representation of sequence.
        """
        _, _ = sequence, agg_fn
        raise NotImplementedError("Four track not yet supported.")

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Embed sequence using six track Naive Mamba.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            agg_fn: Currently unused.

        Returns:
            Naive Mamba representation of sequence.
        """
        if agg_fn is not None:
            raise NotImplementedError(
                "Inference currently does not support alternative aggregation."
            )

        ohe_sequence = str_to_ohe(sequence)

        model_input = np.hstack((
            ohe_sequence,
            cds.reshape(-1, 1),
            splice.reshape(-1, 1)
        ))
        model_input_tt = torch.Tensor(model_input).to(self.device).T
        model_input_tt = model_input_tt.unsqueeze(0)

        hidden_states = self.model(model_input_tt)
        embedding = torch.mean(hidden_states, dim=1)

        return embedding
