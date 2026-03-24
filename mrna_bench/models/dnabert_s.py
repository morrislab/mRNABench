from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class DNABERTS(EmbeddingModel):
    """Inference wrapper for DNABERT-S.

    DNABERT-S is a transformer-based DNA foundation model that builds on
    DNABERT-2 to produce species-aware embeddings for genomic sequences.
    It is trained using a contrastive learning objective which encourages
    grouping of DNA sequences from the same species and discourages grouping
    of sequences from different species. DNABERT-S is trained using microbial
    genomic sequences from viruses, fungi, and bacteria.

    Link: https://github.com/MAGICS-LAB/DNABERT_S
    """

    default_version = "dnabert-s"
    valid_versions = ["dnabert-s"]

    def __init__(self, model_version: str, device: torch.device):
        """Initialize DNABERT-S inference wrapper.

        Args:
            model_version: Version of model used; must be "dnabert-s".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use DNABERT-S."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "quietflamingo/dnaberts-no-flashattention",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "quietflamingo/dnaberts-no-flashattention",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(self.device)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using DNABERT-S.

        ALiBi positional encoding allows for arbitrary sequence lengths.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 768)
        """
        _, _ = cds, splice

        toks = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks)[0]

        # Build pooling mask excluding CLS (pos 0) and SEP (last real pos)
        pooling_mask = toks["attention_mask"].clone()
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        # Apply masked aggregation per sequence
        embeddings = []
        for i in range(hidden_states.size(0)):
            mask = pooling_mask[i].bool()
            masked_hidden = hidden_states[i][mask]
            embeddings.append(agg_fn(masked_hidden))

        return embeddings
