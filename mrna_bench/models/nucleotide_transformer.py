from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class NucleotideTransformer(EmbeddingModel):
    """Inference wrapper for NucleotideTransformer.

    NucleotideTransformer is a transformer based DNA foundation model
    pre-trained using MLM on a variety of pre-training datasets ranging from
    the 1000 (human) genomes project to a multi-species dataset, across
    a wide range of parameters. Input is tokenized to 6-mers where possible,
    with a maximum tokenized sequence length of 1000.

    Link: https://github.com/instadeepai/nucleotide-transformer
    """

    default_version = "2.5b-multi-species"
    valid_versions = [
        "2.5b-multi-species",
        "2.5b-1000g",
        "500m-human-ref",
        "500m-1000g",
        "v2-50m-multi-species",
        "v2-100m-multi-species",
        "v2-250m-multi-species",
        "v2-500m-multi-species",
    ]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "nt-" + model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NucleotideTransformer inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions are: {
                "2.5b-multi-species",
                "2.5b-1000g",
                "500m-human-ref",
                "500m-1000g",
                "v2-50m-multi-species",
                "v2-100m-multi-species",
                "v2-250m-multi-species",
                "v2-500m-multi-species"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
        except ImportError:
            raise ImportError((
                "Install base_models optional dependency to use "
                "NucleotideTransformer."
            ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(self.device)

        self.max_length = self.tokenizer.model_max_length

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask
            excludes padding and special tokens (CLS/SEP).
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        torch_outs = self.model(
            toks["input_ids"],
            attention_mask=toks["attention_mask"],
            encoder_attention_mask=toks["attention_mask"],
            output_hidden_states=True
        )

        hidden_states = torch_outs["hidden_states"][-1]
        pooling_mask = toks["attention_mask"].clone()

        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using NucleotideTransformer.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (1, hidden_dim)
        """
        _, _ = cds, splice

        # NT uses 6-mer tokenization. max_length is in tokens.
        # Each token covers ~6 nucleotides, so max nucleotides is approx
        # (max_length - 2) * 6
        max_chunk_length = (self.max_length - 2) * 6

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=max_chunk_length,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
