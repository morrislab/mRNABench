from collections.abc import Callable

import torch

from multimolecule import RnaTokenizer, RiNALMoModel

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RiNALMo(EmbeddingModel):
    """Inference wrapper for RiNALMo.

    RiNALMo is a transformer-based RNA foundation model trained on 36M ncRNA
    sequences using MLM and other modern architectural improvements such as
    RoPE, SwiGLU activations, and Flash Attention.

    Link: https://github.com/lbcb-sci/RiNALMo

    This wrapper uses the multimoleule implementation of RiNALMo:
    https://huggingface.co/multimolecule/rinalmo
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RiNALMo inference wrapper.

        Args:
            model_version: Version of model to load. Only "rinalmo" valid.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/rinalmo",
            cache_dir=get_model_weights_path()
        )

        self.model = RiNALMoModel.from_pretrained(
            "multimolecule/rinalmo",
            cache_dir=get_model_weights_path()
        ).to(device)

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using RiNALMo.

        Args:
            sequence: Sequence to be embedded.
            overlap: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            RiNALMo embedding of sequence with shape (1 x 1280).
        """
        if overlap != 0:
            raise ValueError("RiNALMo does not require sequence chunking.")

        sequence_in = sequence.replace("T", "U")
        toks = self.tokenizer(sequence_in, return_tensors="pt").to(self.device)

        hidden_output = self.model(**toks).last_hidden_state
        output = agg_fn(hidden_output, dim=1)

        return output

    def embed_sequence_sixtrack(self, sequence, cds, splice, overlap, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for NT.")
