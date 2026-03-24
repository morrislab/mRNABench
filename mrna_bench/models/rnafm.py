from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


class RNAFM(EmbeddingModel):
    """Inference Wrapper for RNA-FM.

    RNA-FM is a transformer based RNA foundation model pre-trained using MLM on
    23 million ncRNA sequences. The primary competency for RNA-FM is ncRNA
    property and structural prediction.

    Link: https://github.com/ml4bio/RNA-FM/
    """

    default_version = "rna-fm"
    valid_versions = ["rna-fm"]

    max_length = 1024

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNA-FM Model.

        Args:
            model_version: Version of RNA-FM to use. Must be "rna-fm".
            device: PyTorch device used by model inference.
        """
        super().__init__(model_version, device)

        try:
            import fm
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNA-FM."
            )

        import os
        hub_path = os.path.join(get_model_weights_path(), "hub")
        old_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(hub_path)

        model, alphabet = fm.pretrained.rna_fm_t12()

        torch.hub.set_dir(old_hub_dir)

        self.model = model.to(device)
        self.batch_converter = alphabet.get_batch_converter()

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        The fm library's batch_converter pads sequences to the longest in the
        batch using the alphabet's padding_idx token.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask) tensors.
        """
        data = [("", chunk) for chunk in chunks]
        _, _, tokens = self.batch_converter(data)

        model_output = self.model(tokens.to(self.device), repr_layers=[12])
        hidden_states = model_output["representations"][12]

        batch_size, seq_len, _ = hidden_states.shape
        pooling_mask = torch.zeros(batch_size, seq_len, device=self.device)
        for i, chunk in enumerate(chunks):
            pooling_mask[i, 1:len(chunk) + 1] = 1

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using RNA-FM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 640)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
