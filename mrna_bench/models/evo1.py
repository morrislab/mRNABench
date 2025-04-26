from collections.abc import Callable

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


    Link: https://github.com/evo-design/evo
    """

    max_length = 8_192

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

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
        self.tokenizer = evo_model.tokenizer.tokenize

        class IdentityEmbedding(nn.Module):
            def unembed(self, u):
                return u

        # need to return the embedding, not logits
        self.model.unembed = IdentityEmbedding()

        if model_version == "evo-1-131k-base":
            self.max_length = 131_072

        revert_model_cache_var(old_hf_cache)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using Evo1.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo1 embedding of sequence with shape (1 x H).
        """
        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        with torch.inference_mode():

            for i, chunk in enumerate(chunks):

                input_ids = torch.tensor(
                    self.tokenizer(chunk),
                    dtype=torch.int
                ).unsqueeze(0).to(self.device)

                # (batch, length, embed dim)
                embeddings, _ = self.model(input_ids)

                embedding_chunks.append(embeddings)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice):
        """Not supported."""
        raise NotImplementedError("Six track not possible with Evo1.")
