from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models.embedding_model import EmbeddingModel


class RNAFM(EmbeddingModel):
    """Inference Wrapper for RNA-FM.

    RNA-FM is a transformer based RNA foundation model pre-trained using MLM on
    23 million ncRNA sequences. The primary competency for RNA-FM is ncRNA
    property and structural prediction.

    mRNA-FM is a related model that is instead pre-trained on coding sequences.
    It can only accept CDS regions (input must be multiple of 3).

    Link: https://github.com/ml4bio/RNA-FM/
    """

    MAX_LENGTH = 1024

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RNA-FM Model.

        Args:
            model_version: Version of RNA-FM to use. Valid versions are:
                {"rna-fm", "mrna-fm"}.
            device: PyTorch device used by model inference.
        """
        super().__init__(model_version, device)

        import fm

        if model_version == "rna-fm":
            model, alphabet = fm.pretrained.rna_fm_t12()
            self.is_sixtrack = False
        elif model_version == "mrna-fm":
            model, alphabet = fm.pretrained.mrna_fm_t12()
            self.is_sixtrack = True
        else:
            raise ValueError("Unknown model version.")

        self.model = model.to(device).eval()
        self.batch_converter = alphabet.get_batch_converter()

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using RNA-FM.

        Due to RNA-FM's max context being shorter than most mRNAs, chunking is
        used. Here, sequence is chunked, and start / end tokens are stripped
        from the middle sequences. Representations are then averaged across
        the sequence length dimension.

        Args:
            sequence: Sequence to embed.
            overlap: Number of overlapping nucleotides between chunks.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            RNA-FM representation of sequence with shape (1 x 640).
        """
        sequence = sequence.replace("U", "T")
        chunks = self.chunk_sequence(sequence, self.MAX_LENGTH - 2, overlap)

        embedding_chunks = []

        for i, chunk in enumerate(chunks):
            _, _, tokens = self.batch_converter([("", chunk)])

            if i == 0:
                tokens = tokens[:, :-1]
            elif i == len(chunks) - 1:
                tokens = tokens[:, 1:]
            else:
                tokens = tokens[:, 1:-1]

            model_output = self.model(tokens.to(self.device), repr_layers=[12])
            embedded_chunk = model_output["representations"][12]

            embedding_chunks.append(embedded_chunk)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        overlap: int,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequence using mRNA-FM.

        Since mRNA-FM only accepts CDS, uses CDS track to extract CDS sequence
        and generate representation from it. CDS sequence must be a multiple
        of three, complicating chunking with overlap. It is disabled for now.

        Args:
            sequence: Sequence to embed.
            cds: Binary encoding of first nucleotide of each codon in CDS.
            splice: Binary encoding of splice site locations.
            overlap: Number of overlapping nucleotides between chunks.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            mRNA-FM representation of CDS of sequence with shape (1 x H).
        """
        if overlap != 0:
            raise ValueError("mRNA-FM wrapper does not support overlap.")

        _ = splice, overlap  # unused

        sequence = sequence.replace("U", "T")

        first_one_index = np.argmax(cds == 1)
        last_one_index = (len(cds) - 1 - np.argmax(np.flip(cds) == 1)) + 2

        cds_seq = sequence[first_one_index:last_one_index + 1]

        if len(cds_seq) % 3 != 0:
            raise ValueError("Length of CDS is not a multiple of 3.")

        chunks = self.chunk_sequence(cds_seq, 1022 * 3)

        embedding_chunks = []

        for i, chunk in enumerate(chunks):
            _, _, tokens = self.batch_converter([("", chunk)])

            if i == 0:
                tokens = tokens[:, :-1]
            elif i == len(chunks) - 1:
                tokens = tokens[:, 1:]
            else:
                tokens = tokens[:, 1:-1]

            model_output = self.model(tokens.to(self.device), repr_layers=[12])
            embedded_chunk = model_output["representations"][12]

            embedding_chunks.append(embedded_chunk)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding
