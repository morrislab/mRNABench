from typing import Callable

import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.utils import get_model_weights_path


class RNABERT(EmbeddingModel):
    """Inference Wrapper for RNABERT.

    RNABERT is a transformer based RNA foundation model pre-trained using
    both a MLM and structural alignment learning objective. RNABERT is
    pre-trained on 80K ncRNA sequences.

    Link: https://github.com/mana438/RNABERT

    This wrapper uses the RNABERT implementation from the multimolecule:
    https://huggingface.co/multimolecule/rnabert
    """

    max_length = 440

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNABERT inference wrapper.

        Args:
            model_version: Must be "rnabert".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaTokenizer, RnaBertModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNABERT."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.model = RnaBertModel.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        ).to(device)

        self.is_sixtrack = False

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed RNA sequence using RNABERT.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            RNABERT embedding of sequence with shape (1 x 120).
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
        """Not implemented for RNABERT."""
        raise NotImplementedError("RNABERT does not support sixtrack mode.")
