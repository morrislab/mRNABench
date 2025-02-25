import math
from functools import partial
import os
import json
import torch
import torch.nn as nn
import numpy as np

from huggingface_hub import PyTorchModelHubMixin

from torch.utils.checkpoint import checkpoint

try:
    from hydra import Hydra
    from mamba_ssm.modules.mamba_simple import Mamba#, Block #- comment in for earlier mamba version
    from mamba_ssm.modules.block import Block # comment out for earlier mamba version
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    from mamba_ssm.modules.mamba_simple import Mamba, Block

class BiMamba(nn.Module):
    """Caduceus wrapper around Mamba to support bi-directionality.
    
    Link: https://github.com/kuleshov-group/caduceus/
    """

    def __init__(
            self,
            d_model: int,
            bidirectional_strategy: str = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")

        self.bidirectional_strategy = bidirectional_strategy
        
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )

        self.mamba_rev = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
            self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
            self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
            self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
            self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)

        out_rev = self.mamba_rev(
            hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
            inference_params=inference_params
        ).flip(dims=(1,))  # Flip back for combining with forward hidden states
        
        if self.bidirectional_strategy == "add":
            out = out + out_rev
        elif self.bidirectional_strategy == "ew_multiply":
            out = out * out_rev
        else:
            raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")

        return out

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bidirectional=None,
    bidirectional_strategy="add",
    bidirectional_weight_tie=True,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if bidirectional is None:
        mix_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    elif bidirectional == "hydra":
        mix_cls = partial(Hydra, layer_idx=layer_idx, use_mem_eff_path=True, **ssm_cfg, **factory_kwargs)
    elif bidirectional == "caduceus":
        mix_cls = partial(BiMamba, layer_idx=layer_idx, bidirectional_strategy=bidirectional_strategy, bidirectional_weight_tie=bidirectional_weight_tie, **ssm_cfg, **factory_kwargs)
    else:
        raise NotImplementedError(f"Bi-directionality strategy `{bidirectional}` not implemented!")
        
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    
    try:
        block = Block(
            d_model,
            mix_cls,
            mlp_cls=nn.Identity, # comment out for earlier mamba version
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    except TypeError:
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
        bidirectional: str = None,
        bidirectional_strategy: str = "add",
        bidirectional_weight_tie: bool = True,
        gradient_checkpointing: bool = False,
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

        self.gradient_checkpointing = gradient_checkpointing

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
                    bidirectional=bidirectional,
                    bidirectional_strategy=bidirectional_strategy,
                    bidirectional_weight_tie=bidirectional_weight_tie,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def _forward_layer(self, layer, hidden_states, residual, inference_params):
        return layer(hidden_states, residual, inference_params=inference_params)

    def forward(self, x, inference_params=None, channel_last=False):
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        residual = None
        for layer in self.layers:

            if self.gradient_checkpointing: # apply gradient checkpointing
                hidden_states, residual = checkpoint(
                    self._forward_layer,
                    layer,
                    hidden_states,
                    residual,
                    inference_params,
                    use_reentrant=False
                )
            else:
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        return hidden_states

    def representation(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Get global representation of input data.

        Args:
            x: Data to embed. Has shape (B x C x L) if not channel_last.
            lengths: Unpadded length of each data input.
            channel_last: Expects input of shape (B x L x C).

        Returns:
            Global representation vector of shape (B x H).
        """
        out = self.forward(x, channel_last=channel_last)

        mean_tensor = mean_unpadded(out, lengths)
        return mean_tensor


def mean_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Take mean of tensor across second dimension without padding.

    Args:
        x: Tensor to take unpadded mean. Has shape (B x L x H).
        lengths: Tensor of unpadded lengths. Has shape (B)

    Returns:
        Mean tensor of shape (B x H).
    """
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked_tensor = x * mask.unsqueeze(-1)
    sum_tensor = masked_tensor.sum(dim=1)
    mean_tensor = sum_tensor / lengths.unsqueeze(-1).float()

    return mean_tensor


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


def load_model(run_path: str, checkpoint_name: str) -> nn.Module:
    """Load trained model located at specified path.

    Args:
        run_path: Path where run data is located.
        checkpoint_name: Name of model checkpoint to load.

    Returns:
        Model with loaded weights.
    """
    model_config_path = os.path.join(run_path, "model_config.json")
    data_config_path = os.path.join(run_path, "data_config.json")

    with open(model_config_path, "r") as f:
        model_params = json.load(f)

    # TODO: Temp backwards compatibility
    if "n_tracks" not in model_params:
        with open(data_config_path, "r") as f:
            data_params = json.load(f)
        n_tracks = data_params["n_tracks"]
    else:
        n_tracks = model_params["n_tracks"]

    model_path = os.path.join(run_path, checkpoint_name)

    # get model name from run_path
    model_name = os.path.basename(run_path)

    model = MixerModel(
        d_model = model_params["ssm_model_dim"],
        n_layer = model_params["ssm_n_layers"],
        input_dim = n_tracks,
        bidirectional = "caduceus" if "caduceus" in model_name else "hydra" if "hydra" in model_name else None,
        bidirectional_strategy = model_params["bidirectional_strategy"] if "bidirectional_strategy" in model_params else "add",
        bidirectional_weight_tie = model_params["bidirectional_weight_tie"] if "bidirectional_weight_tie" in model_params else True,
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict[k.lstrip("model")[1:]] = v

    model.load_state_dict(state_dict)
    return model