from collections.abc import Callable
from functools import partial

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class mRNABERT(EmbeddingModel):
    """Inference wrapper for mRNABERT.

    mRNABERT is a transformer-based RNA foundation model trained on
    36M mRNA sequences using MLM (BERT-style) pre-training. It uses
    ALiBi positional embeddings and Flash Attention for efficient training.
    It was further pre-trained using contrastive learning to align the
    model's CDS embeddings with the corresponding protein embeddings from
    the ProtT5-XL-UniRef50 model.

    Link: https://github.com/yyly6/mRNABERT
    """

    # the model uses ALiBi so it can support length extension
    # during inference, but we empirically set a max length for
    # chunking to avoid OOM issues because transformers have
    # quadratic memory usage with respect to sequence length
    max_length = 10_000

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize mRNABERT inference wrapper.

        Args:
            model_version: Version of model to load.
                Only "mRNABERT" is supported.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel, AutoTokenizer
            from transformers.models.bert.configuration_bert import BertConfig
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use mRNABERT."
            )

        config = BertConfig.from_pretrained(
            "Taykhoom/mRNABERT-no-flashattention",
            cache_dir=get_model_weights_path()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Taykhoom/mRNABERT-no-flashattention",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "Taykhoom/mRNABERT-no-flashattention",
            trust_remote_code=True,
            config=config,
            cache_dir=get_model_weights_path()
        ).to(device).eval()

        self.is_sixtrack = True

    def embed_sequence(self, sequence, agg_fn):
        """Not supported."""
        raise NotImplementedError("Four track not available for mRNABERT.")

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using mRNABERT.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS and the 5' ends of each splice site.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Splice site track for sequence to embed.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            mRNABERT embedding of sequence with shape (1 x 768).
        """
        _ = splice  # unused

        chunks = self.chunk_sequence_cds_aware(
            sequence,
            cds,
            self.max_length - 2,
        )

        embedding_chunks = []

        for chunk_seq, chunk_cds in chunks:
            chunk = self.separate_utr_cds(chunk_seq, chunk_cds)

            toks = self.tokenizer.batch_encode_plus(
                chunk,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]

            last_hidden_state = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]  # [1, chunk_length + padding, hidden_size]

            # REMOVE ZEROED TOKENS #
            # [chunk_length + padding]
            valid = attention_mask.bool()[0]

            # [1, true_chunk_length, hidden_size]
            chunk_embed = last_hidden_state[0, valid, :].unsqueeze(0)

            # # MASK OUT ZEROED TOKENS #
            # mask = attention_mask.unsqueeze(-1) # [1, chunk_len + pad, 1]
            # chunk_embed = last_hidden_state * mask # zero out padding tokens

            embedding_chunks.append(chunk_embed)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding)
        return aggregate_embedding

    def chunk_sequence_cds_aware(
        self,
        sequence: str,
        cds: np.ndarray,
        chunk_length: int
    ) -> list[tuple[str, np.ndarray]]:
        """Chunk sequence while respecting codon boundaries.

        Args:
            sequence: Full RNA sequence.
            cds: CDS track for sequence.
            chunk_length: Maximum length of each chunk.
        Returns:
            List of (sequence chunk, cds chunk) tuples.
        """
        starts = np.where(cds != 0)[0]
        codon_starts = set(starts.tolist())

        if not codon_starts:
            return [
                (
                    sequence[i: i + chunk_length],
                    cds[i: i + chunk_length]
                )
                for i in range(0, len(sequence), chunk_length)
            ]

        chunks = []
        i = 0
        n = len(sequence)

        while i < n:
            end = min(i + chunk_length, n)

            while end > i and any((end - k) in codon_starts for k in (1, 2)):
                end -= 1

            # if we couldn't find a codon boundary, just break at chunk_length
            if end == i:
                end = min(i + chunk_length, n)

            chunks.append((sequence[i:end], cds[i:end]))
            i = end

        return chunks

    def separate_utr_cds(
        self,
        sequence: str,
        cds: np.ndarray,
    ) -> list[str]:
        """Add spacing to separate UTR and CDS regions.

        mRNABERT requires that the UTRs have single character separation
        and the CDS regions have 3 character separation.

        I.E. for a sequence with UTR-CDS-UTR structure with the following
        sequence: "AACTGCGTG" and CDS track: [0,0,1,0,0,0,0,0,0],
        the returned sequence would be: "A A CTG C G T G"

        Args:
            sequence: Full RNA sequence.
            cds: CDS track for sequence.
        Returns:
            Tuple of (5' UTR, CDS, 3' UTR) with appropriate spacing.
        """
        starts = np.where(cds != 0)[0]
        if len(starts) == 0:
            return [" ".join(sequence),]

        start = starts[0]
        end = min(starts[-1] + 3, len(sequence))

        parts = []

        if start > 0:
            parts.append(" ".join(sequence[:start]))

        parts.append(
            " ".join(
                [
                    sequence[i: i + 3]
                    for i in range(start, end, 3)
                ]
            )
        )

        if end < len(sequence):
            parts.append(" ".join(sequence[end:]))

        return [" ".join(parts)]
