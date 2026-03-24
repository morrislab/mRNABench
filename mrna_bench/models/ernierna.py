from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.utils import get_model_weights_path


class ERNIERNA(EmbeddingModel):
    """Inference Wrapper for ERNIE-RNA.

    ERNIE-RNA is a transformer based RNA foundation model pre-trained using
    MLM on 20M ncRNA sequences. ERNIE-RNA uses a custom attention map bias
    based on structural AU / GC / GU pairs.

    A version of ERNIE-RNA which has been fine-tuned on secondary structure
    prediction is available as `ernierna-ss`.

    Link: https://github.com/Bruce-ywj/ERNIE-RNA

    This wrapper uses the ERNIE-RNA implementation from the multimolecule:
    https://huggingface.co/multimolecule/ernierna
    """

    default_version = "ernierna"
    valid_versions = ["ernierna", "ernierna-ss"]

    max_length = 1022  # 1024 - 2 for CLS/SEP

    def __init__(self, model_version: str, device: torch.device):
        """Initialize ERNIE-RNA inference wrapper.

        Args:
            model_version: Version of ERNIE-RNA to use. Valid versions are:
                {"ernierna", "ernierna-ss"}.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import ErnieRnaModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use ERNIE-RNA."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.model = ErnieRnaModel.from_pretrained(
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
        """Embed sequences using ERNIE-RNA.

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
            max_chunk_length=self.max_length,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
