from collections.abc import Callable
import warnings
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


class MRNAFM(EmbeddingModel):
    """Inference Wrapper for mRNA-FM.

    mRNA-FM is a transformer based RNA foundation model pre-trained on coding
    sequences. It can only accept CDS regions (input must be multiple of 3).

    Link: https://github.com/ml4bio/RNA-FM/
    """

    default_version = "mrna-fm"
    valid_versions = ["mrna-fm"]

    max_length = 1024  # in tokens (codons)

    def __init__(self, model_version: str, device: torch.device):
        """Initialize mRNA-FM Model.

        Args:
            model_version: Version of mRNA-FM to use. Must be "mrna-fm".
            device: PyTorch device used by model inference.
        """
        super().__init__(model_version, device)

        try:
            import fm
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use mRNA-FM."
            )

        import os
        hub_path = os.path.join(get_model_weights_path(), "hub")
        old_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(hub_path)

        model, alphabet = fm.pretrained.mrna_fm_t12()

        torch.hub.set_dir(old_hub_dir)

        self.model = model.to(device)
        self.batch_converter = alphabet.get_batch_converter()

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        The fm library's batch_converter pads sequences to the longest in the
        batch using the alphabet's padding_idx token.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask) tensors.
        """
        data = [("", chunk) for chunk in chunks]
        _, _, tokens = self.batch_converter(data)

        model_output = self.model(tokens.to(self.device), repr_layers=[12])
        hidden_states = model_output["representations"][12]

        batch_size, seq_len, _ = hidden_states.shape
        pooling_mask = torch.zeros(batch_size, seq_len, device=self.device)
        for i, chunk in enumerate(chunks):
            num_codons = len(chunk) // 3
            pooling_mask[i, 1:num_codons + 1] = 1

        return hidden_states, pooling_mask

    def get_cds(self, sequence: str, cds: np.ndarray) -> str:
        """Get CDS region of sequence.

        CDS must be a multiple of three. For anomalous sequences, returns as
        much of the CDS as possible that is still a multiple of three.

        Args:
            sequence: Sequence to extract CDS region from.
            cds: Binary encoding of first nucleotide of each codon in CDS.

        Returns:
            Sequence of CDS. Returns original sequence if no CDS found with
            truncation to multiple of three.
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
        """Embed sequences using mRNA-FM.

        Since mRNA-FM only accepts CDS, uses CDS track to extract CDS sequence
        and generate representation from it. CDS sequence must be a multiple
        of three.

        Args:
            sequences: List of sequences to embed.
            cds: List of binary encodings of first nucleotide of each codon.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
                - default (mean): (1280,)
        """
        _ = splice

        if cds is None:
            raise ValueError("mRNA-FM requires cds to extract coding region.")

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
