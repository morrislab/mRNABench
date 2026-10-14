from collections.abc import Callable
from functools import partial

import math

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool
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
                    "Install base_models optional dependency "
                    "to use NaiveMamba."
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
    default_attn_implementation = None

    # mamba_ssm applies in_proj and dt_proj by their weights, bypassing
    # LoRA; x_proj and out_proj are adapted once the fast path is off.
    lora_target_modules = ["x_proj", "out_proj"]
    valid_attn_implementations = None
    hookable_layer_patterns = [r"layers\.\d+"]
    final_layer_pattern = r"norm_f"
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize NaiveMamba model.

        Args:
            model_version: Unused.
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        _ = model_version
        super().__init__(
            "naive-mamba",
            device,
            attn_implementation
        )

        torch.random.manual_seed(0)
        np.random.seed(0)
        self.model = MixerModel(
            d_model=64,
            n_layer=3,
            input_dim=6,
        ).to(device)

    def tokenize(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """One-hot encode sequences with their cds and splice tracks.

        Args:
            sequences: Nucleotide sequences.
            cds: CDS tracks (required).
            splice: Splice site tracks (required).
            add_special_tokens: Unused; NaiveMamba has no special tokens.

        Returns:
            "one_hot" model input and its "attention_mask".
        """
        _ = add_special_tokens
        if cds is None or splice is None:
            raise ValueError("NaiveMamba requires cds and splice tracks.")
        one_hot = [
            torch.as_tensor(
                str_to_ohe(sequence, cds=seq_cds, splice=seq_splice),
                dtype=torch.float32,
            )
            for sequence, seq_cds, seq_splice in zip(sequences, cds, splice)
        ]
        lengths = torch.tensor([len(x) for x in one_hot])
        attention_mask = torch.arange(int(lengths.max())) < lengths[:, None]
        return {
            "one_hot": pad_sequence(one_hot, batch_first=True).to(self.device),
            "attention_mask": attention_mask.long().to(self.device),
        }

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
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
             - default (mean): (64,)
        """
        tokens = self.tokenize(sequences, cds, splice)
        hidden_states = self.model(tokens["one_hot"], channel_last=True)
        return [
            agg_fn(hidden[mask])
            for hidden, mask in zip(hidden_states, self.pooling_mask(tokens))
        ]

    def extract(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        layers: list[int | str] | None = None,
        return_attentions: bool = False,
        offload_to_cpu: bool = True,
    ) -> tuple[
        dict[str, list[list[torch.Tensor]]],
        dict[str, list[list[torch.Tensor]] | None],
    ]:
        """Extract per-layer representations from NaiveMamba.

        Uses forward hooks on Mamba layers. Scores are None for all layers
        (Mamba SSM, no attention weights). Requires cds and splice tracks.

        Args:
            sequences: RNA/DNA sequences.
            cds: CDS tracks (required).
            splice: Splice site tracks (required).
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Ignored (no attention scores for Mamba).
            offload_to_cpu: Move tensors to CPU after each sequence.

        Returns:
            (hidden_states, scores); scores are all None.
        """
        if cds is None or splice is None:
            raise ValueError("NaiveMamba requires cds and splice tracks.")
        resolved = self._resolve_layer_paths(layers)
        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }
        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: None for p in resolved
        }
        for seq, seq_cds, seq_splice in zip(sequences, cds, splice):
            tokens = self._tokenize_one(seq, seq_cds, seq_splice)
            _, activations = self._run_with_layer_capture(
                resolved,
                lambda: self.model(tokens["one_hot"], channel_last=True),
            )
            for path in resolved:
                h = activations[path][0]
                if h.dim() == 3:
                    h = h[0]  # (T, D)
                hidden_out[path].append([h.cpu() if offload_to_cpu else h])

        return hidden_out, score_out
