from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RNAErnie(EmbeddingModel):
    """Inference Wrapper for RNAErnie.

    RNAErnie is a transformer based RNA foundation model pre-trained using
    MLM 23M ncRNA sequences. RNAErnie uses motif-level masking during its
    pre-training by masking contiguous token regions of several sizes.

    Link: https://github.com/CatIIIIIIII/RNAErnie

    This wrapper uses the RNAErnie implementation from the multimolecule:
    https://huggingface.co/multimolecule/rnaernie
    """

    default_version = "rnaernie"
    valid_versions = ["rnaernie"]

    max_length = 512

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNAErnie inference wrapper.

        Args:
            model_version: Must be "rnaernie".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaErnieModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNAErnie."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.model = RnaErnieModel.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
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
    ) -> torch.Tensor:
        """Embed sequences using RNAErnie.

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
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
