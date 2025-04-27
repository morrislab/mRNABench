from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models import EmbeddingModel

import itertools
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBaseline(EmbeddingModel):
    """Inference wrapper for our naive baseline.

    The naive baseline is a simple model that computes k-mer count,
    GC content, exon length, and exon count for each sequence.
    It is a simple feature extraction model that computes features
    from the sequence and returns them as a tensor.
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    @staticmethod
    def generate_vocab(
        kmer_list: list[int] = [3, 4, 5, 6, 7],
        alphabet: str = "ACGT"
    ) -> list[str]:
        """Generate k-mer vocabulary in the given range.

        Args:
            kmer_list: List of k-mer lengths to generate.
            alphabet: Alphabet to use for generating k-mers.

        Returns:
            List of k-mers in the given range.
        """
        kmers: list[str] = []
        for k in sorted(kmer_list):
            kmers.extend(
                ''.join(p) for p in itertools.product(alphabet, repeat=k)
            )

        return kmers

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NaiveBaseline model.

        Args:
            model_version: Version of NaiveBaseline to load. Valid values are:
            {
                "naive-4-track",
                "naive-6-track"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        self.is_sixtrack = model_version == "naive-6-track"
        self.model = "naive_baseline"
        self.device = device

        self.kmer_vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(3, 7),
            vocabulary=self.generate_vocab(),
            lowercase=False,
        )

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable | None = None
    ) -> torch.Tensor:
        """Embed sequence using four track NaiveBaseline.

        Args:
            sequence: Sequence to embed.
            agg_fn: Currently unused.

        Returns:
            NaiveBaseline representation of sequence.
        """
        if agg_fn is not None:
            raise NotImplementedError(
                "Inference currently does not support alternative aggregation."
            )

        # 1. Compute k-mer count (k=3-7)
        kmer_counts = self.kmer_vectorizer.transform([sequence]).toarray()
        kmer_features = torch.tensor(
            kmer_counts,
            dtype=torch.float32
        )[0].squeeze(0)

        # 2. Compute GC content
        a, t = sequence.count('A'), sequence.count('T')
        c, g = sequence.count('C'), sequence.count('G')

        gc_ratio = (g + c) / (a + c + g + t)  # avoid Ns
        gc_content = torch.tensor(
            gc_ratio,
            dtype=torch.float32
        ).unsqueeze(0)

        embedding = torch.cat(
            (kmer_features, gc_content),
            dim=0
        ).unsqueeze(0)

        return embedding

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Embed sequence using six track NaiveBaseline.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            agg_fn: Currently unused.

        Returns:
            NaiveBaseline representation of sequence.
        """
        if agg_fn is not None:
            raise NotImplementedError(
                "Inference currently does not support alternative aggregation."
            )

        # 1. Compute k-mer count (k=3-7) and 2. Compute GC content
        embedding = self.embed_sequence(sequence).squeeze(0)

        # 3. Compute cds length
        cds_positions = np.where(cds == 1)[0]
        if cds_positions.size == 0:
            cds_length = torch.tensor(
                0.0,
                dtype=torch.float32
            ).unsqueeze(0)
        else:
            # last 1 index + 3 to include the end of the last codon
            cds_end = cds_positions[-1] + 3

            # start of first codon
            cds_start = cds_positions[0]
            cds_length = torch.tensor(
                cds_end - cds_start,
                dtype=torch.float32
            ).unsqueeze(0)

        # 4. Compute exon count
        exon_count = np.sum(splice)
        exon_count = torch.tensor(
            exon_count,
            dtype=torch.float32
        ).unsqueeze(0)

        embedding = torch.cat(
            (embedding, cds_length, exon_count),
            dim=0
        ).unsqueeze(0)

        return embedding
