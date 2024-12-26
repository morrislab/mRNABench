import math
from functools import partial
import os
import json
import torch
import torch.nn as nn
import numpy as np

from mamba_ssm.modules.mamba_simple import Mamba, Block
from huggingface_hub import PyTorchModelHubMixin


# convert to one hot
def seq_to_oh(seq):
    oh = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq):
        if base == 'A':
            oh[i, 0] = 1
        elif base == 'C':
            oh[i, 1] = 1
        elif base == 'G':
            oh[i, 2] = 1
        elif base == 'T':
            oh[i, 3] = 1
    return oh


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


class MixerModel(
    nn.Module,
    PyTorchModelHubMixin,
):

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
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x, inference_params=None, channel_last=False):
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        res = None
        for layer in self.layers:
            hidden_states, res = layer(
                hidden_states, res, inference_params=inference_params
            )

        res = (hidden_states + res) if res is not None else hidden_states
        hidden_states = self.norm_f(res.to(dtype=self.norm_f.weight.dtype))

        hidden_states = hidden_states

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

    model = MixerModel(
        d_model=model_params["ssm_model_dim"],
        n_layer=model_params["ssm_n_layers"],
        input_dim=n_tracks
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict[k.lstrip("model")[1:]] = v

    model.load_state_dict(state_dict)
    return model
