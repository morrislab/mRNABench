from collections.abc import Callable
from typing import Dict, Tuple, Any
from functools import partial

import numpy as np
import torch

from mrna_bench import set_model_cache_var, revert_model_cache_var
from mrna_bench.models import EmbeddingModel


# Copied from Evo2 to add forward hook for arbitrary layer output extraction
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
                "use Evo.model.state_dict().keys() to see available layers."
            )

        def hook_fn(layer_name):
            def hook(_, __, output):
                if isinstance(output, tuple):
                    output = output[0]
                embeddings[layer_name] = output  # .detach()
            return hook

        # Register hooks for requested layers
        for name in layer_names:
            layer = self.model.get_submodule(name)
            handles.append(layer.register_forward_hook(hook_fn(name)))

    try:
        # # Original forward pass
        # with torch.no_grad():
        #     logits = self.model.forward(input_ids)
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


def __call__(
    self,
    input_ids: torch.Tensor,
    return_embeddings: bool = False,
    layer_names: list[str] | None = None
):
    return self.forward(input_ids, return_embeddings, layer_names)


class Evo1(EmbeddingModel):
    """Inference wrapper for Evo1.

    Evo1 is a StripedHyena-based DNA foundation model trained on the
    OpenGenome dataset using an autoregressive scheme at single nucleotide,
    byte level resolution. Owing to its StripedHyena backbone, it has a near
    linear scaling of compute and memory relative to its context window.

    Note: StripedHyena's convolutions don't fully isolate sequences in batched
    mode even with padding mask. This implementation uses single-sequence
    processing for consistency.

    Link: https://github.com/evo-design/evo
    """

    default_version = "evo-1.5-8k-base"
    valid_versions = [
        "evo-1.5-8k-base",
        "evo-1-8k-base",
        "evo-1-131k-base",
    ]

    # Evo1 has 32 layers
    # https://github.com/evo-design/evo/issues/32
    # 12th layer embedding has high divergence from final layer
    # so is likely to capture different information
    version_to_middle_layer = {
        "evo-1.5-8k-base": "blocks.12.pre_norm",
        "evo-1-8k-base": "blocks.12.pre_norm",
        "evo-1-131k-base": "blocks.12.pre_norm",
    }

    max_length = 8_192

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Evo1.

        Args:
            model_version: Version of model used. Valid versions: {
                "evo-1.5-8k-base",
                "evo-1-8k-base",
                "evo-1-131k-base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            old_hf_cache = set_model_cache_var()
            from evo import Evo

            # Monkey patch Evo to add layer output extraction via forward hook.
            Evo.forward = forward
            Evo.parameters = parameters
            Evo.__call__ = __call__

        except ImportError:
            revert_model_cache_var(old_hf_cache)
            raise ImportError("Evo must be installed to use this model.")

        self.model = Evo(model_version, device)
        self.tokenizer: Any = self.model.tokenizer

        # we will only take the middle and last layer output for simplicity
        self.embedding_layers = [
            # self.version_to_middle_layer[model_version],
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

        if model_version == "evo-1-131k-base":
            self.max_length = 131_072

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
        """Embed a single sequence using Evo1.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding.

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
                chunk[layer_name].to(self.device)
                for chunk in embedding_chunks
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
        """Embed sequences using Evo1.

        Processes sequences one at a time due to StripedHyena's architectural
        limitation with padding (convolutions don't fully isolate sequences).

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 4096)
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding.squeeze(0))

        return all_embeddings
