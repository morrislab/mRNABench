from collections.abc import Callable
from functools import partial

import torch
import numpy as np

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

    default_version = "v2-eukaryote-3b-base"
    valid_versions = [
        "eukaryote-1.2b-base",
        "v2-eukaryote-1.2b-base",
        "v2-prokaryote-1.2b-base",
        "eukaryote-3b-base",
        "v2-eukaryote-3b-base",
        "v2-prokaryote-3b-base",
    ]

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
            clean_up_tokenization_spaces=True,
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

        self.max_length = 98_304  # based on technical report

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and special tokens (CLS/SEP).
        """
        toks = self.tokenizer(
            chunks,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        ).to(self.device)

        special_tokens_mask = toks.pop("special_tokens_mask")

        hidden_states = self.model(
            **toks,
            output_hidden_states=True
        ).hidden_states[-1]

        pooling_mask = 1 - special_tokens_mask

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using GENERator.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (2048,) for 1.2b models
            - default (mean): (3072,) for 3b models
        """
        _, _ = cds, splice

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
