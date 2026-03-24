from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from torch import nn

from mrna_bench import set_model_cache_var, revert_model_cache_var
from mrna_bench.models import EmbeddingModel


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
        except ImportError:
            revert_model_cache_var(old_hf_cache)
            raise ImportError("Evo must be installed to use this model.")

        evo_model = Evo(model_version)
        self.model = evo_model.model.to(device)
        self._char_tokenizer = evo_model.tokenizer
        self.tokenizer = evo_model.tokenizer.tokenize

        class IdentityEmbedding(nn.Module):
            def unembed(self, u):
                return u

        # need to return the embedding, not logits
        self.model.unembed = IdentityEmbedding()

        # PEFT compatibility: config.to_dict is None, PEFT expects callable
        # Provide method that returns config as dictionary
        if hasattr(self.model, 'config') and not callable(getattr(self.model.config, 'to_dict', None)):
            config_dict = dict(self.model.config)
            self.model.config.to_dict = lambda: config_dict

        if model_version == "evo-1-131k-base":
            self.max_length = 131_072

        revert_model_cache_var(old_hf_cache)

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
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo1 embedding with shape (1, 4096).
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        for chunk in chunks:
            input_ids = torch.tensor(
                self.tokenizer(chunk),
                dtype=torch.int,
            ).unsqueeze(0).to(self.device)

            embeddings, _ = self.model(input_ids)
            chunk_embedding = agg_fn(embeddings[0], dim=0)
            embedding_chunks.append(chunk_embedding)

        if len(embedding_chunks) == 1:
            return embedding_chunks[0].unsqueeze(0).float()

        all_chunks = torch.stack(embedding_chunks, dim=0)
        return agg_fn(all_chunks).unsqueeze(0).float()

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Embed sequences using Evo1.

        Processes sequences one at a time due to StripedHyena's architectural
        limitation with padding (convolutions don't fully isolate sequences).

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo1 embeddings with shape (batch_size, 4096).
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding)

        return torch.cat(all_embeddings, dim=0)
