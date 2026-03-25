from collections.abc import Callable
from typing import Dict, Tuple, Any
from functools import partial
import os

import numpy as np
import torch

from mrna_bench import set_model_cache_var, revert_model_cache_var
from mrna_bench.models.embedding_model import EmbeddingModel


# Copied from Evo2, removed .detach()/torch.no_grad() for gradient flow
# https://github.com/ArcInstitute/evo2/blob/main/evo2/models.py#L59
def forward(
    self,
    input_ids: torch.Tensor,
    return_embeddings: bool = False,
    layer_names: list[str] | None = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
    """
    Forward pass with optional embedding extraction.

    Args:
        input_ids: Input token IDs
        return_embeddings: If True, returns embeddings from specified layers
        layer_names: List of layer names to extract embeddings from.
            Required if return_embeddings=True

    Returns:
        Tuple of (logits, embeddings_dict) if return_embeddings=True
        Tuple of (logits, None) otherwise
    """
    embeddings = {}
    handles = []

    if return_embeddings:
        if layer_names is None:
            raise ValueError(
                "layer_names must be specified when return_embeddings=True. "
                "use Evo2.model.state_dict().keys() to see available layers."
            )

        def hook_fn(layer_name):
            def hook(_, __, output):
                if isinstance(output, tuple):
                    output = output[0]
                embeddings[layer_name] = output
            return hook

        # Register hooks for requested layers
        for name in layer_names:
            layer = self.model.get_submodule(name)
            handles.append(layer.register_forward_hook(hook_fn(name)))

    try:
        logits = self.model.forward(input_ids)

        if return_embeddings:
            return logits, embeddings
        return logits, None

    finally:
        for handle in handles:
            handle.remove()


def parameters(self, recurse: bool = True):
    """Override parameters to ensure gradients are tracked."""
    return self.model.parameters(recurse)


class Evo2(EmbeddingModel):
    """Inference wrapper for Evo2.

    Evo2 is a StripedHyena2-based DNA foundation model trained on the
    OpenGenome2 dataset using an autoregressive scheme at single nucleotide
    resolution. Owing to its StripedHyena2 backbone, it has an ultra long
    context window. The `base` variants can handle sequences up to 8192
    nucleotides in length while the larger variants can handle sequences up
    to 1 million nucleotides in length. While it can in principle handle
    sequences longer than 1 MB due, due to GPU memory constraints, we limit
    the maximum sequence length to 1,000,000 nucleotides. This can be
    increased if more GPU memory is available.

    Link: https://github.com/ArcInstitute/evo2
    """

    default_version = "evo2_7b"
    valid_versions = [
        "evo2_1b_base",
        "evo2_7b_base",
        "evo2_7b",
        "evo2_20b",
        "evo2_40b_base",
        "evo2_40b",
    ]

    max_length = 8_192
    version_to_middle_layer = {
        "evo2_1b_base": "blocks.12.pre_norm",
        "evo2_7b_base": "blocks.16.pre_norm",
        "evo2_7b": "blocks.16.pre_norm",
        "evo2_20b": "blocks.12.pre_norm",
        "evo2_40b_base": "blocks.25.pre_norm",
        "evo2_40b": "blocks.25.pre_norm",
    }

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Evo2.

        Args:
            model_version: Version of model used. Valid versions: {
                "evo2_40b",
                "evo2_20b",
                "evo2_7b",
                "evo2_7b_262k",
                "evo2_40b_base",
                "evo2_7b_base",
                "evo2_1b_base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        # On HPC systems with multiple CUDA installations, TransformerEngine's
        # NVRTC JIT compiler may pick up system CUDA headers (e.g. from
        # /usr/local/cuda-*) instead of the conda env's CUDA headers,
        # causing compilation failures. NVTE_CUDA_INCLUDE_DIR tells TE which
        # headers to pass to NVRTC.
        # Users can override by setting NVTE_CUDA_INCLUDE_DIR in their env.
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix and not os.environ.get("NVTE_CUDA_INCLUDE_DIR"):
            cuda_inc = os.path.join(
                conda_prefix, "targets", "x86_64-linux", "include"
            )
            if os.path.isfile(os.path.join(cuda_inc, "cuda_fp8.hpp")):
                os.environ["NVTE_CUDA_INCLUDE_DIR"] = cuda_inc

        try:
            old_hf_cache = set_model_cache_var()
            from evo2 import Evo2 as Evo2Model

            Evo2Model.forward = forward
            Evo2Model.parameters = parameters

            # Patch TransformerEngine's FP8 amax aggregation to handle
            # multi-GPU sharding. TE registers amax (absolute-maximum)
            # buffers from each FP8 layer on that layer's device; torch.cat
            # fails when those tensors span cuda:0 and cuda:1. We move all
            # amax tensors to the device of the first entry before the
            # concatenation so that the reduction proceeds normally.
            import transformer_engine.pytorch.fp8 as _te_fp8
            _mgr = _te_fp8.FP8GlobalStateManager
            _orig_reduce = (
                _mgr.reduce_and_update_fp8_tensors.__func__
            )

            @classmethod  # type: ignore[misc]
            def _multi_device_safe_reduce(cls, forward: bool = True) -> None:
                for key in list(cls.global_amax_buffer.keys()):
                    buf = cls.global_amax_buffer[key]
                    if len(buf) > 1:
                        target_device = buf[0].device
                        cls.global_amax_buffer[key] = [
                            t.to(target_device) for t in buf
                        ]
                _orig_reduce(cls, forward=forward)

            _mgr.reduce_and_update_fp8_tensors = _multi_device_safe_reduce
        except ImportError:
            revert_model_cache_var(old_hf_cache)
            raise ImportError("Evo2 must be installed to use this model.")

        self.model = Evo2Model(model_version)
        self.tokenizer: Any = self.model.tokenizer

        # Evo2's StripedHyena2 SSM filter layers (HyenaCascade) store
        # their poles and residues as nn.Parameters created under
        # torch.inference_mode() because the model config sets
        # inference_mode=True. Inference tensors cannot be saved for
        # backward, causing autograd-tracked forward passes to fail.
        # Replace any inference-mode Parameters with fresh clones so
        # they become regular autograd-compatible Parameters.
        inner: torch.nn.Module = (
            self.model.model  # type: ignore[assignment, union-attr]
        )
        for module in inner.modules():
            for param_name, param in list(
                module.named_parameters(recurse=False)
            ):
                if param is not None and param.is_inference():
                    new_param = torch.nn.Parameter(
                        param.data.clone(),
                        requires_grad=param.requires_grad
                    )
                    module.register_parameter(param_name, new_param)

        # we will only take the middle and last layer output for simplicity
        self.embedding_layers = [
            self.version_to_middle_layer[model_version],
            'norm'
        ]

        # PEFT compatibility: config.to_dict is None, PEFT expects callable
        # Provide method that returns config as dictionary
        if (
            hasattr(self.model, 'config') and not
            callable(getattr(self.model.config, 'to_dict', None))
        ):
            config_dict = dict(self.model.config)  # type: ignore[arg-type]
            self.model.config.to_dict = (  # type: ignore[union-attr]
                lambda: config_dict
            )

        # NOTE:
        # - https://github.com/ArcInstitute/evo2/issues/160
        # - https://github.com/ArcInstitute/evo2/issues/172
        # While non-base 7B+ versions of Evo2 can theoretically handle
        # sequences >=262K nts, we find that getting embeddings for sequences
        # longer than 75K nt can be problematic for embedding on a single GPU
        # (see issue details). If this is encountered, we manually reduce the
        # max length. Long-term, the solution is to use multi-GPU inference
        # for these very long sequences, but this isn't currently implemented.
        if model_version in ["evo2_40b", "evo2_7b"]:
            self.max_length = 1_000_000

        if model_version == "evo2_7b_262k":
            self.max_length = 262_144

        revert_model_cache_var(old_hf_cache)

    def set_inference_mode(self):
        """Set model to inference mode with gradients disabled."""
        self.model.model.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        """Set model to training mode with gradients enabled."""
        self.model.model.train()
        torch.set_grad_enabled(True)

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Embed a single sequence using Evo2.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Tensor representing embedded sequence.
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        toks = self.tokenizer.tokenize_batch(chunks)

        for tok_chunks in toks:

            input_ids = torch.tensor(
                tok_chunks,
                dtype=torch.int
            ).unsqueeze(0).to(self.device)

            _, embeddings = self.model(
                input_ids=input_ids,
                return_embeddings=True,
                layer_names=self.embedding_layers
            )
            embedding_chunks.append(embeddings)

        aggregate_embeddings = []
        for layer_name in sorted(self.embedding_layers):
            layer_chunks = [
                chunk[layer_name].to(self.device) for chunk in embedding_chunks
            ]
            agg_chunks = agg_fn(
                torch.cat(layer_chunks, dim=1).squeeze(0)
            )
            aggregate_embeddings.append(
                agg_chunks.float().unsqueeze(0)
            )

        return torch.cat(aggregate_embeddings, dim=1)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using Evo2.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            List of embeddings with item shape depending on agg_fn.
            - default (mean): (hidden_dim * num_layers,)
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            emb = self.embed_sequence(sequence, agg_fn=agg_fn).squeeze(0)
            all_embeddings.append(emb)

        return all_embeddings
