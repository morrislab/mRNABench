from typing import Callable
from functools import partial

import math

import numpy as np

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block


from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.datasets.dataset_utils import str_to_ohe


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mix_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mix_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        input_dim: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Linear(input_dim, d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                self._init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x, inference_params=None, channel_last=False):
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(
            residual.to(dtype=self.norm_f.weight.dtype)
        )

        hidden_states = hidden_states

        return hidden_states

    @staticmethod
    def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
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
    """Naive Mamba model which uses default initialization."""

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Orthrus model.

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
        """Embed sequence using four track Orthrus.

        Args:
            sequence: Sequence to embed.
            agg_fn: Currently unused.

        Returns:
            Orthrus representation of sequence.
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
        """Embed sequence using six track Orthrus.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            agg_fn: Currently unused.

        Returns:
            Orthrus representation of sequence.
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
