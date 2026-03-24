from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class CodonBERT(EmbeddingModel):
    """Inference wrapper for CodonBERT.

    CodonBERT is a transformer-based RNA language model that is
    pretrained on more than 10 million mRNA sequences from mammals,
    bacteria, and human viruses using MLM. It is specifically trained
    on coding regions of mRNA sequences, and is designed for predicting
    mRNA-specific properties.

    Link: https://github.com/Sanofi-Public/CodonBERT
    """

    default_version = "codonbert"
    valid_versions = ["codonbert"]

    max_length = 1024  # in tokens (codons)
    max_length_nt = (1024 - 2) * 3  # in nucleotides, accounting for CLS/SEP

    def __init__(self, model_version: str, device: torch.device):
        """Initialize CodonBERT inference wrapper.

        Args:
            model_version: Version of model used; must be "codonbert".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use CodonBERT."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "lhallee/CodonBERT",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "lhallee/CodonBERT",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        ).to(self.device)

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
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state
        pooling_mask = toks["attention_mask"].clone()

        # Exclude special tokens (CLS at pos 0, SEP at last real pos)
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
    ) -> torch.Tensor:
        """Embed sequences using CodonBERT.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with shape (batch_size, 768).
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length_nt,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
