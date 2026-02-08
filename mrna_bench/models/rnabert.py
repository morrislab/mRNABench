from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.utils import get_model_weights_path


class RNABERT(EmbeddingModel):
    """Inference Wrapper for RNABERT.

    RNABERT is a transformer based RNA foundation model pre-trained using
    both a MLM and structural alignment learning objective. RNABERT is
    pre-trained on 80K ncRNA sequences.

    Link: https://github.com/mana438/RNABERT

    This wrapper uses the RNABERT implementation from the multimolecule:
    https://huggingface.co/multimolecule/rnabert
    """

    default_version = "rnabert"
    valid_versions = ["rnabert"]

    max_length = 440

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNABERT inference wrapper.

        Args:
            model_version: Must be "rnabert".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaTokenizer, RnaBertModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNABERT."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.model = RnaBertModel.from_pretrained(
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
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using RNABERT.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with shape (batch_size, 120).
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
