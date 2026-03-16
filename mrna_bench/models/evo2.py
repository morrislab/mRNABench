from collections.abc import Callable
from functools import partial

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
    to 1 million nucleotides in length. While it can in principle handle
    sequences longer than 1 MB due, due to GPU memory constraints, we limit
    the maximum sequence length to 1,000,000 nucleotides. This can be
    increased if more GPU memory is available.

    Link: https://github.com/ArcInstitute/evo2
    """

    max_length = 8_192
    version_to_middle_layer = {
        "evo2_40b": "blocks.25.mlp.l3",
        "evo2_20b": "blocks.12.mlp.l3",
        "evo2_7b": "blocks.16.mlp.l3",
        "evo2_7b_262k": "blocks.16.mlp.l3",
        "evo2_40b_base": "blocks.25.mlp.l3",
        "evo2_7b_base": "blocks.16.mlp.l3",
        "evo2_1b_base": "blocks.12.mlp.l3",
    }

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-")

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

        try:
            old_hf_cache = set_model_cache_var()
            from evo2 import Evo2 as Evo2Model
        except ImportError:
            revert_model_cache_var(old_hf_cache)
            raise ImportError("Evo2 must be installed to use this model.")

        self.model = Evo2Model(model_version)
        self.tokenizer = self.model.tokenizer.tokenize

        # we will only take the middle and last layer output for simplicity
        self.embedding_layers = [
            self.version_to_middle_layer[model_version],
            'norm'
        ]

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

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using Evo2.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo2 embedding of sequence with shape (1 x H).
        """
        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        with torch.inference_mode():

            for i, chunk in enumerate(chunks):

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

        # embedding is of type bfloat16, need to convert to float32
        # since numpy does not support bfloat16
        for layer_name in sorted(self.embedding_layers):
            n_chunks = len(embedding_chunks)
            layer_chunks = [
                embedding_chunks[i][layer_name] for i in range(n_chunks)
            ]
            agg_chunks = agg_fn(torch.cat(layer_chunks, dim=1))
            aggregate_embeddings.append(agg_chunks.float().cpu())

        # concatenate the embeddings across the layers
        aggregate_embedding = torch.cat(aggregate_embeddings, dim=1)

        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not possible with Evo2.")
