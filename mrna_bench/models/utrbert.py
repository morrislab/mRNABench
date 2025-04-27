from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class UTRBERT(EmbeddingModel):
    """Inference wrapper for 3UTRBERT.

    3UTRBERT is a transformer-based mRNA foundation model pretrained on the
    3'UTR regions of 100k RNA sequences from gencode using MLM. Various
    versions of 3UTRBERT are available with different k-mer sizes (3, 4, 5, 6).

    This wrapper also offers the option to only use the 3'UTR region of the
    sequence. This can be specified by adding '-utronly' to the end of the
    model version.

    Link: https://github.com/yangyn533/3UTRBERT

    This wrapper uses the multimolecule implementation of 3UTRBERT:
    https://huggingface.co/multimolecule/utrbert-3mer
    """

    max_length = 512

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize 3UTRBERT.

        Args:
            model_version: Version of model to load. Valid versions: {
                "utrbert-3mer",
                "utrbert-4mer",
                "utrbert-5mer",
                "utrbert-6mer",
            }. Add '-utronly' to end of version to only use 3'UTR region.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from multimolecule import RnaTokenizer, UtrBertModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use 3UTRBERT."
            )

        if "-utronly" in model_version:
            self.is_sixtrack = True
        else:
            self.is_sixtrack = False

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/{}".format(model_version.replace("-utronly", "")),
            cache_dir=get_model_weights_path()
        )
        self.model = UtrBertModel.from_pretrained(
            "multimolecule/{}".format(model_version.replace("-utronly", "")),
            cache_dir=get_model_weights_path()
        ).to(device)

        self.kmer_size = int(model_version.split("-")[1].replace("mer", ""))

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using 3UTRBERT.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            3UTRBERT embedding of sequence with shape (1 x 768).
        """
        sequence = sequence.replace("T", "U")
        tokens = self.tokenizer(sequence, return_tensors="pt")
        input_ids = list(tokens["input_ids"][0][1:-1])

        chunks = self.chunk_tokens(input_ids, self.max_length - 2)

        embedding_chunks = []

        for chunk in chunks:
            chunk_ids = torch.Tensor([[1] + chunk + [2]])
            chunk_attn = torch.Tensor([[1] * (len(chunk) + 2)])

            toks = {
                "input_ids": chunk_ids.long().to(self.device),
                "attention_mask": chunk_attn.long().to(self.device)
            }

            cls_output = self.model(**toks).last_hidden_state[:, 0:1, :]
            embedding_chunks.append(cls_output)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using only 3'UTR region using 3UTRBERT.

        Args:
            sequence: Sequence to be embedded.
            cds: CDS indices.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            3UTRBERT embedding of sequence with shape (1 x 768).
        """
        _ = splice  # Unused
        cds_sequence = self.get_threeprime_utr(sequence, cds)
        return self.embed_sequence(cds_sequence, agg_fn)

    def get_threeprime_utr(self, sequence: str, cds: np.ndarray) -> str:
        """Get 3'UTR region of sequence.

        Args:
            sequence: Sequence to extract 3'UTR from.
            cds: CDS indices.

        Returns:
            3'UTR region of sequence, or full sequence if no CDS found.
        """
        if sum(cds) == 0:
            return sequence

        cds_end = np.where(cds == 1)[0][-1] + 3

        if len(sequence[cds_end:]) < self.kmer_size:
            return sequence

        return sequence[cds_end:]
