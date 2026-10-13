from collections.abc import Callable
import itertools

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

from mrna_bench.models import EmbeddingModel, ModelBehavior


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
    kmer_vectorizer: CountVectorizer,
    features: tuple[str, ...] = ("kmer", "gc"),
) -> torch.Tensor:
    """Compute k-mer counts and GC content for a sequence.

    Args:
        sequence: DNA sequence to compute features for.
        kmer_vectorizer: CountVectorizer for k-mer counting.
        features: Features to compute, from "kmer" and "gc".

    Returns:
        Tensor of the selected features with shape (num_features,).
    """
    parts = [torch.zeros(0)]
    if "kmer" in features:
        kmer_counts = kmer_vectorizer.transform([sequence]).toarray()
        parts.append(torch.tensor(kmer_counts[0], dtype=torch.float32))

    if "gc" in features:
        a, t = sequence.count("A"), sequence.count("T")
        c, g = sequence.count("C"), sequence.count("G")
        gc_ratio = (g + c) / (a + c + g + t)  # avoid Ns
        parts.append(torch.tensor([gc_ratio], dtype=torch.float32))

    return torch.cat(parts, dim=0)


class NaiveBaseline(EmbeddingModel):
    """Inference wrapper for naive baseline (4-track).

    Computes k-mer counts (k=3-7) and GC content from the sequence. The
    ``-kmer`` and ``-gc`` versions compute only that feature.
    """

    version_features = {
        "naive-4-track": ("kmer", "gc"),
        "naive-4-track-kmer": ("kmer",),
        "naive-4-track-gc": ("gc",),
    }
    default_version = "naive-4-track"
    valid_versions = list(version_features)
    default_attn_implementation = None
    valid_attn_implementations = None
    hookable_layer_patterns = []
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
        kmer_list: list[int] = [3, 4, 5, 6, 7],
    ):
        """Initialize NaiveBaseline model.

        Args:
            model_version: Version of NaiveBaseline to load.
            device: PyTorch device (unused, kept for interface consistency).
            attn_implementation: Attention backend (unused, kept
                for interface consistency).
            kmer_list: List of k-mer lengths to count.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        self.model = torch.nn.Identity()
        self.features = self.version_features[model_version]
        if list(kmer_list) != [3, 4, 5, 6, 7]:
            self.model_options = {"kmer_list": list(kmer_list)}
        self.kmer_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(min(kmer_list), max(kmer_list)),
            vocabulary=generate_kmer_vocab(kmer_list),
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
            embedding = compute_kmer_gc_features(
                seq, self.kmer_vectorizer, self.features
            )
            embeddings.append(embedding)

        return embeddings


class NaiveBaselineSixTrack(EmbeddingModel):
    """Inference wrapper for naive baseline (6-track).

    Computes k-mer counts (k=3-7), GC content, CDS length, and exon count.
    The other versions compute a subset; ``struct`` is CDS length and exon
    count.
    """

    version_features = {
        "naive-6-track": ("kmer", "gc", "cds", "exon"),
        "naive-6-track-kmer": ("kmer",),
        "naive-6-track-struct": ("cds", "exon"),
        "naive-6-track-gc-struct": ("gc", "cds", "exon"),
    }
    default_version = "naive-6-track"
    valid_versions = list(version_features)
    default_attn_implementation = None
    valid_attn_implementations = None
    hookable_layer_patterns = []
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
        kmer_list: list[int] = [3, 4, 5, 6, 7],
    ):
        """Initialize NaiveBaselineSixTrack model.

        Args:
            model_version: Version of NaiveBaselineSixTrack to load.
            device: PyTorch device (unused, kept for interface consistency).
            attn_implementation: Attention backend (unused, kept
                for interface consistency).
            kmer_list: List of k-mer lengths to count.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        self.model = torch.nn.Identity()
        self.features = self.version_features[model_version]
        if list(kmer_list) != [3, 4, 5, 6, 7]:
            self.model_options = {"kmer_list": list(kmer_list)}
        self.kmer_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(min(kmer_list), max(kmer_list)),
            vocabulary=generate_kmer_vocab(kmer_list),
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
             - default (mean): (n_features,)
        """
        _ = agg_fn

        if cds is None or splice is None:
            raise ValueError(
                "NaiveBaselineSixTrack requires cds and splice tracks."
            )

        embeddings = []
        for i, sequence in enumerate(sequences):
            parts = [compute_kmer_gc_features(
                sequence,
                self.kmer_vectorizer,
                self.features,
            )]
            if "cds" in self.features:
                cds_positions = np.where(cds[i] == 1)[0]
                cds_length = 0
                if cds_positions.size > 0:
                    cds_length = cds_positions[-1] + 3 - cds_positions[0]
                parts.append(torch.tensor([cds_length], dtype=torch.float32))
            if "exon" in self.features:
                parts.append(
                    torch.tensor([np.sum(splice[i])], dtype=torch.float32)
                )
            embeddings.append(torch.cat(parts, dim=0))

        return embeddings
