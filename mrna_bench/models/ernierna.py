from typing import Callable

import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.utils import get_model_weights_path


class ERNIERNA(EmbeddingModel):
    """Inference Wrapper for ERNIE-RNA.

    ERNIE-RNA is a transformer based RNA foundation model pre-trained using
    MLM on 20M ncRNA sequences. ERNIE-RNA uses a custom attention map bias
    based on structural AU / GC / GU pairs.

    A version of ERNIE-RNA which has been fine-tuned on secondary structure
    prediction is available as `ernierna-ss`.

    Link: https://github.com/Bruce-ywj/ERNIE-RNA

    This wrapper uses the ERNIE-RNA implementation from the multimolecule:
    https://huggingface.co/multimolecule/ernierna
    """

    max_length = 1024

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize ERNIE-RNA inference wrapper.

        Args:
            model_version: Version of ERNIE-RNA to use. Valid versions are:
                {"ernierna", "ernierna-ss"}.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import ErnieRnaModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use ERNIE-RNA."
            )

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        )

        self.model = ErnieRnaModel.from_pretrained(
            "multimolecule/{}".format(model_version),
            cache_dir=get_model_weights_path()
        ).to(device)

        self.is_sixtrack = False

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using ERNIE-RNA.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            ERNIE-RNA embedding of sequence with shape (1 x 768).
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
        """Not implemented for ERNIE-RNA."""
        raise NotImplementedError("ERNIE-RNA does not support sixtrack mode.")
