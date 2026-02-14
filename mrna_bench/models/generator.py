from collections.abc import Callable
from functools import partial

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class GENERator(EmbeddingModel):
    """Inference wrapper for GENERator.

    GENERator is a Transformer-based autoregressive genomic foundation model
    using k-mer tokenization. The original model is trained with standard
    next-token prediction on gene-centric functional regions to enable
    long-context generative modeling of eukaryotic genomes.

    GENERator-v2 retains the same backbone and tokenization but introduces
    Factorized Nucleotide Supervision (FNS), which decomposes each k-mer
    prediction into nucleotide-level likelihoods, and Genome Compression
    Pretraining (GCP), which concatenates functional regions to densify
    biological signal and induce next-gene prediction. v2 supports contexts
    up to 98k base pairs and includes eukaryotic and prokaryotic variants.

    Link: https://github.com/GenerTeam/GENERator
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "GENERator-" + model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize GENERator inference wrapper.

        Args:
            model_version: Version of GENERator to load. Valid values are: {
                "eukaryote-1.2b-base",
                "eukaryote-3b-base",
                "v2-eukaryote-1.2b-base",
                "v2-eukaryote-3b-base",
                "v2-prokaryote-1.2b-base",
                "v2-prokaryote-3b-base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use GENERator."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "GenerTeam/GENERator-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "GenerTeam/GENERator-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(self.device)

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        # Set pad_token to eos_token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = self.model.config
        self.max_length = config.max_position_embeddings

    def right_pad_sequence(self, sequence: str,) -> str:
        """Right pad sequence to be multiple of 6 in length."""
        padding_length = (6 - (len(sequence) % 6)) % 6
        return sequence + ("A" * padding_length)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using GENERator.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            GENERator embedding of sequence with shape (1 x H).
            H is:
                2048 for 1.2b models
                3072 for 3b models
        """
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        with torch.inference_mode():
            for i, chunk in enumerate(chunks):

                # GENERator needs input sequence length to be multiple of 6
                chunk = self.right_pad_sequence(chunk)

                inputs = self.tokenizer(
                    chunk,
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                if len(chunks) != 1:
                    if i == 0:
                        inputs = {k: v[:, :-1] for k, v in inputs.items()}
                    elif i == len(chunks) - 1:
                        inputs = {k: v[:, 1:] for k, v in inputs.items()}
                    else:
                        inputs = {k: v[:, 1:-1] for k, v in inputs.items()}

                model_out = self.model(
                    **inputs,
                    output_hidden_states=True
                ).hidden_states[-1]

                embedding_chunks.append(model_out)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for GENERator.")
