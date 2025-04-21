from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class UTRLM(EmbeddingModel):
    """Inference wrapper for UTR-LM.

    UTR-LM is a transformer-based mRNA foundation model that is pre-trained on
    random and endogenous 5'UTR sequences from various species using MLM.

    Link: https://github.com/a96123155/UTR-LM

    This wrapper uses the multimoleule implementation of UTR-LM:
    https://multimolecule.danling.org/models/utrlm/

    It is unclear from the manuscript what the max token input is, so the value
    from multimolecule's version is used (accounting for cls/sep tokens).

    This model also offers the ability to predict solely on the 5'UTR,
    which is in-distribution for the model training data. This can be specified
    by adding '-utr' to the end of the model version.
    """

    max_length = 1026

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize UTR-LM inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions: {
                "utrlm-te_el",
                "utrlm-mrl",
                "utrlm-te_el-utronly",
                "utrlm-mrl-utronly"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import UtrLmModel, RnaTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use UTR-LM."
            )

        if "-utronly" in model_version:
            self.is_sixtrack = True
        else:
            self.is_sixtrack = False

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version.replace("-utronly", "")),
            cache_dir=get_model_weights_path()
        )

        self.model = UtrLmModel.from_pretrained(
            "multimolecule/{}".format(model_version.replace("-utronly", "")),
            cache_dir=get_model_weights_path()
        ).to(device)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using UTR-LM.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            UTR-LM embedding of sequence with shape (1 x 128).
        """
        sequence = sequence.replace("T", "U")
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer(chunk, return_tensors="pt").to(self.device)

            chunk_output = self.model(**toks).last_hidden_state

            embedding_chunks.append(chunk_output)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed ONLY the 5'UTR of a sequence using UTR-LM.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Unused.
            agg_fn: Currently unused.

        Returns:
            UTR-LM representation of 5'UTR of the sequence.
        """
        _ = splice
        five_utr_seq = self.get_fiveprime_utr(sequence, cds)

        return self.embed_sequence(five_utr_seq, agg_fn)

    def get_fiveprime_utr(self, sequence: str, cds: np.ndarray) -> str:
        """Return the portion of a sequence corresponding to the 5'UTR.

        If no CDS is detected or entire sequence is CDS, return orignal input.

        Args:
            sequence: Sequence to process.
            cds: Binary array denoting CDS.

        Returns:
            Sequence's 5'UTR, or original sequence if UTR cannot be found.
        """
        if sum(cds) == 0 or len(sequence[:np.argmax(cds)]) == 0:
            return sequence
        return sequence[:np.argmax(cds)]
