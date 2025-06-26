import os
import torch
import numpy as np
from mrna_bench.models import EmbeddingModel
from collections.abc import Callable
from mrna_bench.datasets.dataset_utils import str_to_ohe
from orthrus_src import load_model
from saluki_src import load_saluki_model
from dilated_resnet_src import load_dilated_resnet_model

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
        self.model.eval()

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

class Saluki(EmbeddingModel):

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
        
        if "medium" in model_version:
            model_size = "medium"
        else:
            model_size = "small"

        model = load_saluki_model(
            f"{os.path.join(model_repository, '')}{model_version}",
            checkpoint_name=checkpoint,
            model_size=model_size
        )

        self.is_sixtrack = True

        self.model = model.to(device)
        self.model.eval()

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("Saluki is a six-track model, use embed_sequence_sixtrack.")

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        overlap: int = 0,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Embed sequence using six track Saluki.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            overlap: Unused.
            agg_fn: Currently unused.

        Returns:
            Saluki representation of sequence.
        """
        if overlap != 0:
            raise ValueError("Saluki does not chunk sequence.")

        if agg_fn is not None:
            raise NotImplementedError(
                "Inference currently does not support alternative aggregation."
            )

        MIN_LEN = 320
        if len(sequence) < MIN_LEN:
            pad_len = MIN_LEN - len(sequence)
            sequence += "N" * pad_len
            cds = np.pad(cds, (0, pad_len))
            splice = np.pad(splice, (0, pad_len))

        ohe_sequence = str_to_ohe(sequence)

        model_input = np.hstack((
            ohe_sequence,
            cds.reshape(-1, 1),
            splice.reshape(-1, 1)
        ))

        model_input_tt = torch.Tensor(model_input).to(self.device)
        model_input_tt = model_input_tt.transpose(0, 1) # LxC -> CxL
        model_input_tt = model_input_tt.unsqueeze(0) # BxCxL

        embedding = self.model.representation(
            model_input_tt
        )

        return embedding

class DilatedResnetBench(EmbeddingModel):

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

        model = load_dilated_resnet_model(
            f"{os.path.join(model_repository, '')}{model_version}",
            checkpoint_name=checkpoint,
            model_version=model_version
        )

        self.is_sixtrack = True
        self.model = model.to(device)
        self.model.eval()

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("DilatedResnet is a six-track model, use embed_sequence_sixtrack.")

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        overlap: int = 0,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Embed sequence using six track DilatedResnet.
        """
        if overlap != 0:
            raise ValueError("DilatedResnet does not chunk sequence.")

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
        model_input_tt = model_input_tt.transpose(0, 1) # LxC -> CxL
        model_input_tt = model_input_tt.unsqueeze(0) # BxCxL

        lengths = torch.LongTensor([model_input_tt.shape[2]]).to(self.device)

        embedding = self.model.representation(
            model_input_tt,
            lengths=lengths
        )

        return embedding