from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.datasets.dataset_utils import str_to_ohe
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

    def __init__(self, model_version: str, 
        checkpoint : str, 
        device: torch.device,
        model_repository: str = "/data1/morrisq/ian/rna_contrast/runs/"
        # model_repository: str = "/data1/morrisq/dalalt1/Orthrus/models/"
    ):
        """Initialize Orthrus model.

        Args:
            model_version: Version of Orthrus to load. Valid values are: {
                "orthrus-base-4-track",
                "orthrus-large-6-track"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use Orthrus."
            )
        
        if checkpoint is not None:
            # if 'bidirectional' in model_version:
            #     from mrna_bench.models.cerebrus_src import load_model
            # else:
            from mrna_bench.models.orthrus_src import load_model

            model = load_model(
                f"{model_repository}{model_version}",
                checkpoint_name=checkpoint,
            )

            self.is_sixtrack = '6_track' in model_version or '6t' in model_version

        else: 
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

        embedding = self.model.representation(
            model_input_tt,
            lengths,
            channel_last=True
        )

        return embedding
