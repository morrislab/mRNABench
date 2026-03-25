from collections.abc import Callable
from functools import partial
import warnings

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class CodonBERT(EmbeddingModel):
    """Inference wrapper for CodonBERT.

    CodonBERT is a transformer-based RNA language model that is
    pretrained on more than 10 million mRNA sequences from mammals,
    bacteria, and human viruses using MLM. It is specifically trained
    on coding regions of mRNA sequences, and is designed for predicting
    mRNA-specific properties.

    Link: https://github.com/Sanofi-Public/CodonBERT
    """

    default_version = "codonbert"
    valid_versions = ["codonbert"]

    max_length = 1024  # in tokens (codons)

    def __init__(self, model_version: str, device: torch.device):
        """Initialize CodonBERT inference wrapper.

        Args:
            model_version: Version of model used; must be "codonbert".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use CodonBERT."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "lhallee/CodonBERT",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "lhallee/CodonBERT",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        ).to(self.device)

    @staticmethod
    def _nt_to_codons(sequence: str) -> str:
        """Convert a nucleotide sequence to space-separated codons.

        Only complete codons are included; trailing 1-2 nucs are dropped.
        Sequences should already be in RNA space (U not T).

        Args:
            sequence: Nucleotide sequence (RNA, i.e. using U).

        Returns:
            Space-separated codon string, e.g. "AUG UAA GCA".
        """
        n = len(sequence) - len(sequence) % 3
        return " ".join(sequence[i:i + 3] for i in range(0, n, 3))

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of nucleotide sequence chunks (RNA, i.e. using U).

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and special tokens (CLS/SEP).
        """
        codon_chunks = [self._nt_to_codons(chunk) for chunk in chunks]

        toks = self.tokenizer(
            codon_chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state
        pooling_mask = toks["attention_mask"].clone()

        # Exclude special tokens (CLS at pos 0, SEP at last real pos)
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def get_cds(self, sequence: str, cds: np.ndarray) -> str:
        """Get CDS region of sequence.

        CDS must be a multiple of three. For anomalous sequences, returns as
        much of the CDS as possible that is still a multiple of three.

        Args:
            sequence: Sequence to extract CDS region from (RNA, using U).
            cds: Binary encoding of first nucleotide of each codon in CDS.

        Returns:
            CDS subsequence. Returns truncated full sequence if no CDS found.
        """
        if sum(cds) == 0:
            warnings.warn("No CDS found. Returning truncated sequence.")
            return sequence[:len(sequence) - (len(sequence) % 3)]

        first_one_index = np.argmax(cds == 1)
        last_one_index = (len(cds) - 1 - np.argmax(np.flip(cds) == 1)) + 2

        proposed_cds = sequence[first_one_index:last_one_index + 1]

        if len(proposed_cds) % 3 != 0:
            warnings.warn("Irregular CDS. Returning truncated sequence.")
            return proposed_cds[:-(len(proposed_cds) % 3)]

        return proposed_cds

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using CodonBERT.

        CodonBERT is trained on CDS sequences only. If cds tracks are
        provided, the CDS region is extracted before embedding. Otherwise,
        the full sequence is used (truncated to a multiple of three).

        Args:
            sequences: List of sequences to embed.
            cds: Optional list of binary arrays marking first nucleotide of
                each codon. Used to extract the CDS region from full mRNA.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,)
        """
        _ = splice

        if cds is None:
            raise ValueError(
                "CodonBERT requires cds to extract coding region."
            )

        processed_sequences = []
        for i, seq in enumerate(sequences):
            seq = seq.replace("T", "U")
            seq = self.get_cds(seq, cds[i])
            processed_sequences.append(seq)

        return self._embed_with_chunking(
            sequences=processed_sequences,
            max_chunk_length=(self.max_length - 2) * 3,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
