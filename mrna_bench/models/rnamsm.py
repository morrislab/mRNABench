from typing import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RNAMSM(EmbeddingModel):
    """Inference wrapper for RNA-MSM.

    RNA-MSM is a transformer-based RNA foundation model pretrained using custom
    structure-based MSAs between ~4000 RNA families with ~3000 MSAs each.

    Link: https://github.com/yikunpku/RNA-MSM

    This wrapper uses the multimolecule implementation of RNA-MSM:
    https://huggingface.co/multimolecule/rnamsm

    It is a known issue that the multimolecule implementation of RNA-MSM is
    not fully compatible with the original RNA-MSM implementation. However,
    the discrepancy lies in a missing EOS token and should not significantly
    affect the performance of the model.
    """

    max_length = 1024

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNA-MSM.

        Args:
            model_version: Must be "rnamsm".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaMsmModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNA-MSM."
            )

        self.model = RnaMsmModel.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        ).to(device)

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.is_sixtrack = False

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using RNA-MSM.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            RNA-MSM embedding of sequence with shape (1 x 768).
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
        """Not implemented for RNA-MSM."""
        raise NotImplementedError("RNA-MSM does not support sixtrack mode.")
