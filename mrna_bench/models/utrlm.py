from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class UTRLM(EmbeddingModel):
    """Inference wrapper for UTR-LM.

    UTR-LM is a transformer-based mRNA foundation model that is pre-trained on
    random and endogenous 5'UTR sequences from various species using MLM.

    Link: https://github.com/a96123155/UTR-LM

    This wrapper uses the multimolecule implementation of UTR-LM:
    https://multimolecule.danling.org/models/utrlm/

    It is unclear from the manuscript what the max token input is, so the value
    from multimolecule's version is used (accounting for cls/sep tokens).
    """

    default_version = "utrlm-te_el"
    valid_versions = [
        "utrlm-te_el",
        "utrlm-mrl",
    ]

    max_length = 1026

    def __init__(self, model_version: str, device: torch.device):
        """Initialize UTR-LM inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions: {
                "utrlm-te_el", "utrlm-mrl"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import UtrLmModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use UTR-LM."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            extra_special_tokens={},
            cache_dir=get_model_weights_path()
        )

        self.model = UtrLmModel.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        ).to(device)

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask) tensors.
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state

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
        """Embed sequences using UTR-LM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 128)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
