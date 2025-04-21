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

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Helix-mRNA model.

        Args:
            model_version: Must be "helix-mrna".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from helical import HelixmRNA, HelixmRNAConfig
        except ImportError:
            raise ImportError("Helix-mRNA missing required dependencies.")

        helix_mrna_config = HelixmRNAConfig(
            batch_size=1,
            device=device
        )

        self.model = HelixmRNA(configurer=helix_mrna_config)
        self.is_sixtrack = True

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using Helix-mRNA.

        Args:
            sequence: Sequence to embed.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Helix-mRNA representation of sequence with shape (1 x 256).
        """
        sequence = sequence.upper().replace("T", "U")

        dataset = self.model.process_data(sequence)
        rna_embeddings = torch.Tensor(self.model.get_embeddings(dataset))

        return agg_fn(rna_embeddings, dim=1)

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequence using Helix-mRNA.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Converts sequence to Helix-mRNA vocabulary by inserting an 'E' token
        at the start of every codon.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Helix-mRNA representation of sequence with shape (1 x 256).
        """
        _ = splice  # Unused

        modified_sequence = self.tokenize_cds(sequence, cds)
        embedding = self.embed_sequence(modified_sequence, agg_fn=agg_fn)

        return embedding

    def tokenize_cds(self, sequence: str, cds: np.ndarray) -> str:
        """Convert sequence to Helix-mRNA vocab by inserting 'E' tokens."""
        modified_sequence = ""
        for i in range(len(sequence)):
            if cds[i] == 1:
                modified_sequence += "E"
            modified_sequence += sequence[i]

        return modified_sequence
