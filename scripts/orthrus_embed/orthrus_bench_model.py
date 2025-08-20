import os
import torch
import numpy as np
from mrna_bench.models import EmbeddingModel
from collections.abc import Callable
from mrna_bench.datasets.dataset_utils import str_to_ohe
from orthrus_src import load_model
from orthrus_src import mean_unpadded

class Orthrus(EmbeddingModel):

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-").replace("-track", "").replace("best-", "")

    def __init__(self,
        model_version: str,
        checkpoint : str,
        device: torch.device,
        model_repository: str
    ):

        super().__init__(
                model_version + "_" + checkpoint.replace(".ckpt", ""),
                device
            )

        model = load_model(
            f"{os.path.join(model_repository, '')}{model_version}",
            checkpoint_name=checkpoint,
        )

        self.is_sixtrack = '6_track' in model_version or '6t' in model_version

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

        ohe_sequence = torch.from_numpy(str_to_ohe(sequence)).to(self.device, dtype=torch.float32)
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
        subset_start: int | None = None,
        subset_end: int | None = None,
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

        ohe_sequence = str_to_ohe(sequence)

        model_input = np.hstack((
            ohe_sequence,
            cds.reshape(-1, 1),
            splice.reshape(-1, 1)
        ))

        model_input_tt = torch.Tensor(model_input).to(self.device)
        model_input_tt = model_input_tt.unsqueeze(0)

        lengths = torch.Tensor([model_input_tt.shape[1]]).to(self.device)

        if subset_start is not None:
            pre_mean = self.model.forward(model_input_tt, channel_last=True)

            pre_mean = pre_mean[:, subset_start:subset_end, :]

            embedding = mean_unpadded(pre_mean, lengths)

        else:
            embedding = self.model.representation(
                model_input_tt,
                lengths,
                channel_last=True
            )

        return embedding