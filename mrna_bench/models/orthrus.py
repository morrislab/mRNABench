from collections.abc import Callable

import numpy as np

import torch

from mrna_bench.models.embedding_model import EmbeddingModel

from mrna_bench.datasets.dataset_utils import str_to_ohe


class Orthrus(EmbeddingModel):
    CKPT_DICT = {
        "orthrus_large_6_track": "epoch=22-step=20000.ckpt"
    }

    def __init__(self, model_version, device):
        super().__init__(model_version, device)

        from mrna_bench.models.orthrus_src import load_model

        model_repository = "/home/shir2/mRNABench/model_weights/"
        model = load_model(
            f"{model_repository}{model_version}",
            checkpoint_name=self.CKPT_DICT[model_version]
        )

        if model_version == "orthrus_large_6_track":
            self.is_sixtrack = True
        else:
            self.is_sixtrack = False

        self.model = model.to(device)

    def get_model_short_name(model_version: str) -> str:
        return model_version.replace("_track", "").replace("_", "-")

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        pass

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        overlap: int = 0,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
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
