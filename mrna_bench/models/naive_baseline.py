from collections.abc import Callable
import itertools

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

from mrna_bench.models import EmbeddingModel


def generate_kmer_vocab(
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
            "".join(p) for p in itertools.product(alphabet, repeat=k)
        )
    return kmers


def compute_kmer_gc_features(
    sequence: str,
    kmer_vectorizer: CountVectorizer
) -> torch.Tensor:
    """Compute k-mer counts and GC content for a sequence.

    Args:
        sequence: DNA sequence to compute features for.
        kmer_vectorizer: CountVectorizer for k-mer counting.

    Returns:
        Tensor of k-mer counts and GC content with shape (num_features,).
    """
    kmer_counts = kmer_vectorizer.transform([sequence]).toarray()
    kmer_features = torch.tensor(
        kmer_counts,
        dtype=torch.float32
    )[0].squeeze(0)

    a, t = sequence.count("A"), sequence.count("T")
    c, g = sequence.count("C"), sequence.count("G")
    gc_ratio = (g + c) / (a + c + g + t)  # avoid Ns
    gc_content = torch.tensor(gc_ratio, dtype=torch.float32).unsqueeze(0)

    return torch.cat((kmer_features, gc_content), dim=0)


class NaiveBaseline(EmbeddingModel):
    """Inference wrapper for naive baseline (4-track).

    Computes k-mer counts (k=3-7) and GC content from the sequence.
    """

    default_version = "naive-4-track"
    valid_versions = ["naive-4-track"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NaiveBaseline model.

        Args:
            model_version: Version of NaiveBaseline to load.
            device: PyTorch device (unused, kept for interface consistency).
        """
        super().__init__(model_version, device)

        self.model = torch.nn.Identity()
        self.kmer_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(3, 7),
            vocabulary=generate_kmer_vocab(),
            lowercase=False,
        )

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> list[torch.Tensor]:
        """Embed sequences using NaiveBaseline.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Unused.

        Returns:
            NaiveBaseline embeddings with shape (batch_size, num_features).
        """
        _, _, _ = cds, splice, agg_fn

        embeddings = []
        for seq in sequences:
            embedding = compute_kmer_gc_features(seq, self.kmer_vectorizer)
            embeddings.append(embedding)

        return embeddings


class NaiveBaselineSixTrack(EmbeddingModel):
    """Inference wrapper for naive baseline (6-track).

    Computes k-mer counts (k=3-7), GC content, CDS length, and exon count.
    """

    default_version = "naive-6-track"
    valid_versions = ["naive-6-track"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NaiveBaselineSixTrack model.

        Args:
            model_version: Version of NaiveBaselineSixTrack to load.
            device: PyTorch device (unused, kept for interface consistency).
        """
        super().__init__(model_version, device)

        self.model = torch.nn.Identity()
        self.kmer_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(3, 7),
            vocabulary=generate_kmer_vocab(),
            lowercase=False,
        )

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> list[torch.Tensor]:
        """Embed sequences using NaiveBaselineSixTrack.

        Args:
            sequences: List of sequences to embed.
            cds: CDS tracks for sequences.
            splice: Splice site tracks for sequences.
            agg_fn: Unused.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (1, n_features)
        """
        _ = agg_fn

        if cds is None or splice is None:
            raise ValueError(
                "NaiveBaselineSixTrack requires cds and splice tracks."
            )

        embeddings = []
        for i, sequence in enumerate(sequences):
            base_features = compute_kmer_gc_features(
                sequence,
                self.kmer_vectorizer
            )

            cds_positions = np.where(cds[i] == 1)[0]
            if cds_positions.size == 0:
                cds_length = torch.tensor(
                    0.0,
                    dtype=torch.float32
                ).unsqueeze(0)
            else:
                cds_end = cds_positions[-1] + 3
                cds_start = cds_positions[0]
                cds_length = torch.tensor(
                    cds_end - cds_start,
                    dtype=torch.float32
                ).unsqueeze(0)

            exon_count = torch.tensor(
                np.sum(splice[i]),
                dtype=torch.float32
            ).unsqueeze(0)

            embedding = torch.cat(
                (base_features, cds_length, exon_count),
                dim=0
            )
            embeddings.append(embedding)

        return embeddings
