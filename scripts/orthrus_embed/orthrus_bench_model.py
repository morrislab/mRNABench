import os
import torch
import numpy as np
from mrna_bench.models import EmbeddingModel
from collections.abc import Callable
from mrna_bench.datasets.dataset_utils import str_to_ohe
from functools import partial
from orthrus_src import load_model

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
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using four track Orthrus.

        Args:
            sequence: Sequence to embed.
            agg_fn: Currently unused.

        Returns:
            Orthrus representation of sequence.
        """
        if self.is_sixtrack:
            raise ValueError((
                "Currently loaded model is six track."
                "Use embed_sequence_sixtrack instead."
            ))

        ohe_sequence = torch.from_numpy(str_to_ohe(sequence)).to(self.device, dtype=torch.float32)
        model_input_tt = ohe_sequence.unsqueeze(0)

        embedding = self.model(model_input_tt, channel_last=True)

        aggregated_embedding = agg_fn(embedding)
        return aggregated_embedding

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using six track Orthrus.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            agg_fn: Currently unused.

        Returns:
            Orthrus representation of sequence.
        """
        ohe_sequence = str_to_ohe(sequence)

        model_input = np.hstack((
            ohe_sequence,
            cds.reshape(-1, 1),
            splice.reshape(-1, 1)
        ))

        model_input_tt = torch.Tensor(model_input).to(self.device)
        model_input_tt = model_input_tt.unsqueeze(0)

        embedding = self.model(model_input_tt, channel_last=True)

        aggregated_embedding = agg_fn(embedding)
        return aggregated_embedding
