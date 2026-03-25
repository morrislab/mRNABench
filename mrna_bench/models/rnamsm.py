from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RNAMSM(EmbeddingModel):
    """Inference wrapper for RNA-MSM.

    RNA-MSM is a transformer-based RNA foundation model pretrained using custom
    structure-based MSAs between ~4000 RNA families with ~3000 MSAs each.

    Link: https://github.com/yikunpku/RNA-MSM

    This wrapper uses the multimolecule implementation of RNA-MSM:
    https://huggingface.co/multimolecule/rnamsm

    It is a known issue that the multimolecule implementation of RNA-MSM is
    not fully compatible with the original RNA-MSM implementation. However,
    the discrepancy lies in a missing EOS token and should not significantly
    affect the performance of the model.
    """

    default_version = "rnamsm"
    valid_versions = ["rnamsm"]

    max_length = 1024

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNA-MSM.

        Args:
            model_version: Must be "rnamsm".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaMsmModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNA-MSM."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            extra_special_tokens={},
            cache_dir=get_model_weights_path()
        )

        self.model = RnaMsmModel.from_pretrained(
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
        """Embed sequences using RNA-MSM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
