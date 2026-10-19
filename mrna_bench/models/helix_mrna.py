from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models import EmbeddingModel


class HelixmRNAWrapper(EmbeddingModel):
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
        batch_size: int = 32
    ):
        """Initialize Helix-mRNA model.

        Args:
            model_version: Must be "helix-mrna".
            device: PyTorch device to send model to.
            batch_size: Batch size for inference.
        """
        super().__init__(model_version, device)

        try:
            from helical import HelixmRNA, HelixmRNAConfig
        except ImportError:
            raise ImportError("Helix-mRNA missing required dependencies.")

        helix_mrna_config = HelixmRNAConfig(
            batch_size=batch_size,
            device=device
        )

        self.model = HelixmRNA(configurer=helix_mrna_config)

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
        agg_fn: Callable = torch.mean,
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
            Helix-mRNA embeddings with shape (batch_size, 256).
        """
        _ = splice  # Unused

        if cds is not None:
            sequences = [
                self._tokenize_cds(seq, c).upper().replace("T", "U")
                for seq, c in zip(sequences, cds)
            ]
        else:
            sequences = [s.upper().replace("T", "U") for s in sequences]

        dataset = self.model.process_data(sequences)
        embeddings = torch.Tensor(self.model.get_embeddings(dataset))

        # embeddings shape: (batch, seq_len, hidden)
        return agg_fn(embeddings, dim=1)
