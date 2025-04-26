from typing import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RNAErnie(EmbeddingModel):
    """Inference Wrapper for RNAErnie.

    RNAErnie is a transformer based RNA foundation model pre-trained using
    MLM 23M ncRNA sequences. RNAErnie uses motif-level masking during its
    pre-training by masking contiguous token regions of several sizes.

    Link: https://github.com/CatIIIIIIII/RNAErnie

    This wrapper uses the RNAErnie implementation from the multimolecule:
    https://huggingface.co/multimolecule/rnaernie
    """

    max_length = 512

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNAErnie inference wrapper.

        Args:
            model_version: Must be "rnaernie".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaErnieModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNAErnie."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.model = RnaErnieModel.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        ).to(device)

        self.is_sixtrack = False

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed RNA sequence using RNAErnie.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            RNAErnie embedding of sequence with shape (1 x 768).
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
        """Not implemented for RNAErnie."""
        raise NotImplementedError("RNAErnie does not support sixtrack mode.")
