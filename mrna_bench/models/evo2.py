from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import set_model_cache_var, revert_model_cache_var
from mrna_bench.models.embedding_model import EmbeddingModel


class Evo2(EmbeddingModel):
    """Inference wrapper for Evo2.

    Evo2 is a StripedHyena2-based DNA foundation model trained on the
    OpenGenome2 dataset using an autoregressive scheme at single nucleotide
    resolution. Owing to its StripedHyena2 backbone, it has an ultra long
    context window. The `base` variants can handle sequences up to 8192
    nucleotides in length while the larger variants can handle sequences up
    to 1 million nucleotides in length.

    Link: https://github.com/ArcInstitute/evo2
    """

    default_version = "evo2_7b"
    valid_versions = [
        "evo2_40b",
        "evo2_7b",
        "evo2_40b_base",
        "evo2_7b_base",
        "evo2_1b_base",
    ]

    max_length = 8_192
    version_to_middle_layer = {
        "evo2_40b": "blocks.25.pre_norm",
        "evo2_7b": "blocks.16.pre_norm",
        "evo2_40b_base": "blocks.25.pre_norm",
        "evo2_7b_base": "blocks.16.pre_norm",
        "evo2_1b_base": "blocks.12.pre_norm"
    }

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Evo2.

        Args:
            model_version: Version of model used. Valid versions: {
                "evo2_40b",
                "evo2_7b",
                "evo2_40b_base",
                "evo2_7b_base",
                "evo2_1b_base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            old_hf_cache = set_model_cache_var()
            from evo2 import Evo2
        except ImportError:
            revert_model_cache_var(old_hf_cache)
            raise ImportError("Evo2 must be installed to use this model.")

        self.model = Evo2(model_version)
        self.tokenizer = self.model.tokenizer.tokenize

        # we will only take the middle and last layer output for simplicity
        self.embedding_layers = [
            self.version_to_middle_layer[model_version],
            'norm'
        ]

        if model_version in ["evo2_40b", "evo2_7b"]:
            self.max_length = 1_000_000

        revert_model_cache_var(old_hf_cache)

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed a single sequence using Evo2.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embedding with shape (1, hidden_dim * num_layers).
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        for chunk in chunks:
            input_ids = torch.tensor(
                self.tokenizer(chunk),
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
            layer_chunks = [chunk[layer_name] for chunk in embedding_chunks]
            agg_chunks = agg_fn(torch.cat(layer_chunks, dim=1), dim=1)
            aggregate_embeddings.append(agg_chunks.float())

        return torch.cat(aggregate_embeddings, dim=1)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using Evo2.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo2 embeddings with shape (batch_size, hidden_dim * num_layers).
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            all_embeddings.append(self.embed_sequence(sequence, agg_fn=agg_fn))

        return torch.cat(all_embeddings, dim=0)
