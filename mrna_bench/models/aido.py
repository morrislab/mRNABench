from collections.abc import Callable
from functools import partial

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


class AIDORNA(EmbeddingModel):
    """Inference wrapper for AIDO.RNA.

    AIDO.RNA is a transformer-based RNA foundation model. It is trained using
    masked language modelling on 42 million non-coding RNA sequences, with
    domain adaptation models available for protein coding sequences.

    Link: https://github.com/genbio-ai/ModelGenerator
    """

    max_length = 1024

    default_version = "aido_rna_650m_cds"
    valid_versions = [
        "aido_rna_650m",
        "aido_rna_650m_cds",
        "aido_rna_1b600m",
        "aido_rna_1b600m_cds"
    ]

    def __init__(self, model_version: str, device: torch.device):
        """Initialize AIDO.RNA.

        Args:
            model_version: Version of model used. Valid versions: {
                "aido_rna_1b600m",
                "aido_rna_1b600m_cds",
                "aido_rna_650m",
                "aido_rna_650m_cds",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use AIDO.RNA."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Taykhoom/AIDO-RNA-Wrapper",
            trust_remote_code=True,
            clean_up_tokenization_spaces=True,
            cache_dir=get_model_weights_path(),
        )

        self.model = AutoModel.from_pretrained(
            "Taykhoom/AIDO-RNA-Wrapper",
            trust_remote_code=True,
            base_model=model_version,
            cache_dir=get_model_weights_path(),
        ).to(device)

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
            add_special_tokens=True,
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
    ) -> list[torch.Tensor]:
        """Embed sequences using AIDO.RNA.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1280,) for 650M model
            - default (mean): (2048,) for 1.6B model
        """
        _, _ = cds, splice

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
