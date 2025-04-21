from collections.abc import Callable

import torch

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

    max_length = 8192

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

        try:
            from multimolecule import RnaTokenizer, RiNALMoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RiNALMo."
            )

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
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using RiNALMo.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            RiNALMo embedding of sequence with shape (1 x 1280).
        """
        sequence = sequence.replace("T", "U")
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer(chunk, return_tensors="pt").to(self.device)

            cls_output = self.model(**toks).last_hidden_state
            embedding_chunks.append(cls_output)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for NT.")
