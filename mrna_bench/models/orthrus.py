from collections.abc import Callable

import numpy as np
import torch

from transformers import AutoModel

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class Orthrus(EmbeddingModel):
    """Inference wrapper for Orthrus.

    Orthrus is a RNA foundation model trained using a Mamba backbone. It uses
    a contrastive learning pre-training objective that maximizes similarity
    between RNA splice isoforms and orthologous transcripts. Input length is
    unconstrained due to use of Mamba.

    Link: https://github.com/bowang-lab/Orthrus
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Orthrus model.

        Args:
            model_version: Version of Orthrus to load. Valid values are: {
                "orthrus-base-4-track",
                "orthrus-large-6-track"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        model_hf_path = "quietflamingo/{}".format(model_version)
        model = AutoModel.from_pretrained(
            model_hf_path,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.is_sixtrack = model_version == "orthrus-large-6-track"
        self.model = model.to(device)

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable | None = None
    ) -> torch.Tensor:
        """Embed sequence using four track Orthrus.

        Args:
            sequence: Sequence to embed.
            overlap: Unused.
            agg_fn: Currently unused.

        Returns:
            Orthrus representation of sequence.
        """
        if overlap != 0:
            raise ValueError("Orthrus does not chunk sequence.")

        if agg_fn is not None:
            raise NotImplementedError(
                "Inference currently does not support alternative aggregation."
            )

        ohe_sequence = self.model.seq_to_oh(sequence).to(self.device)
        model_input_tt = ohe_sequence.unsqueeze(0)

        lengths = torch.Tensor([model_input_tt.shape[1]]).to(self.device)

        embedding = self.model.representation(
            model_input_tt,
            lengths,
            channel_last=True
        )

        return embedding

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        overlap: int = 0,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Embed sequence using six track Orthrus.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            overlap: Unused.
            agg_fn: Currently unused.

        Returns:
            Orthrus representation of sequence.
        """
        if overlap != 0:
            raise ValueError("Orthrus does not chunk sequence.")

        if agg_fn is not None:
            raise NotImplementedError(
                "Inference currently does not support alternative aggregation."
            )

        ohe_sequence = self.model.seq_to_oh(sequence).numpy()

        model_input = np.hstack((
            ohe_sequence,
            cds.reshape(-1, 1),
            splice.reshape(-1, 1)
        ))

        model_input_tt = torch.Tensor(model_input).to(self.device)
        model_input_tt = model_input_tt.unsqueeze(0)

        lengths = torch.Tensor([model_input_tt.shape[1]]).to(self.device)

        embedding = self.model.representation(
            model_input_tt,
            lengths,
            channel_last=True
        )

        return embedding
