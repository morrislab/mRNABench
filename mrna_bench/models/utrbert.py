from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class UTRBERT(EmbeddingModel):
    """Inference wrapper for 3UTRBERT.

    3UTRBERT is a transformer-based mRNA foundation model pretrained on the
    3'UTR regions of 100k RNA sequences from gencode using MLM. Various
    versions of 3UTRBERT are available with different k-mer sizes (3, 4, 5, 6).

    Link: https://github.com/yangyn533/3UTRBERT

    This wrapper uses the multimolecule implementation of 3UTRBERT:
    https://huggingface.co/multimolecule/utrbert-3mer
    """

    default_version = "utrbert-6mer"
    valid_versions = [
        "utrbert-3mer",
        "utrbert-4mer",
        "utrbert-5mer",
        "utrbert-6mer",
    ]

    max_length = 512

    def __init__(self, model_version: str, device: torch.device):
        """Initialize 3UTRBERT.

        Args:
            model_version: Version of model to load. Valid versions: {
                "utrbert-3mer", "utrbert-4mer", "utrbert-5mer", "utrbert-6mer"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaTokenizer, UtrBertModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use 3UTRBERT."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )
        self.model = UtrBertModel.from_pretrained(
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
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using 3UTRBERT.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            3UTRBERT embeddings with shape (batch_size, 768).
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
