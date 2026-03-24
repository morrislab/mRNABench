from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class HelixmRNA(EmbeddingModel):
    """Inference wrapper for Helix-mRNA.

    Helix-mRNA is a RNA foundation model trained using a Mamba2 and transformer
    hybrid backbone. Helix-mRNA is pre-trained on 26M mRNAs from diverse
    eukaryotic and viral species.

    Link: https://github.com/helicalAI/helical
    """

    default_version = "helix-mrna"
    valid_versions = ["helix-mrna"]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
    ):
        """Initialize Helix-mRNA model.

        Args:
            model_version: Must be "helix-mrna".
            device: PyTorch device to send model to.
            batch_size: Batch size for inference.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("Helix-mRNA missing required dependencies.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Taykhoom/Helix-mRNA-Wrapper",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "Taykhoom/Helix-mRNA-Wrapper",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(self.device)

        self.max_length = self.tokenizer.model_max_length

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
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_special_tokens_mask=True,
        ).to(self.device)

        special_tokens_mask = toks["special_tokens_mask"]
        attention_mask = 1 - special_tokens_mask

        hidden_states = self.model(
            input_ids=toks["input_ids"],
            attention_mask=attention_mask,
        ).last_hidden_state

        pooling_mask = attention_mask.clone()

        return hidden_states, pooling_mask

    def _tokenize_cds(self, sequence: str, cds: np.ndarray) -> str:
        """Convert sequence to Helix-mRNA vocab by inserting 'E' tokens."""
        modified_sequence = ""
        for i in range(len(sequence)):
            if cds[i] == 1:
                modified_sequence += "E"
            modified_sequence += sequence[i]

        return modified_sequence

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Batch embed sequences using Helix-mRNA.

        If cds is provided, inserts 'E' tokens at the start of each codon
        to use Helix-mRNA's codon-aware vocabulary.

        Args:
            sequences: List of sequences to embed.
            cds: List of binary encodings of first nucleotide of each codon.
            splice: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (1, 256)
        """
        _ = splice  # Unused

        if cds is not None:
            sequences = [
                self._tokenize_cds(seq, c).upper().replace("T", "U")
                for seq, c in zip(sequences, cds)
            ]
        else:
            sequences = [s.upper().replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 1, # Account for SEP token
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
