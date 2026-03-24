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

    default_version = "mRNABERT"
    valid_versions = ["mRNABERT"]

    # the model uses ALiBi so it can support length extension
    # during inference, but we empirically set a max length for
    # chunking to avoid OOM issues because transformers have
    # quadratic memory usage with respect to sequence length
    max_length = 10_000

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
            from transformers.models.bert.modeling_bert import BertModel
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
        ).to(device)

        # The remote code registers a custom BertConfig subclass (with
        # alibi_starting_size) as the AutoModel handler for BertConfig.
        # Reset to the standard BertModel so that subsequent models that
        # use plain AutoModel.from_pretrained on a BertConfig don't pick
        # up the mRNABERT model class and fail on missing config fields.
        AutoModel._model_mapping.register(
            BertConfig,
            (BertModel, BertModel),
            exist_ok=True
        )

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Batch embed sequences using mRNABERT.

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks for sequences. If provided, will trigger
                6-track embedding.
            splice: List of splice tracks for sequences. Unused for mRNABERT.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (1, 768)
        """
        if cds is not None:
            return self.embed_sixtrack(sequences, cds, splice, agg_fn)
        else:
            raise ValueError(
                "CDS tracks must be provided for mRNABERT embedding."
            )

    def _forward_chunks_sixtrack(
        self,
        transformed_chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for batched transformed chunks.

        Args:
            transformed_chunks: List of transformed chunk strings
                (already processed through separate_utr_cds).

        Returns:
            Tuple of (hidden_states, pooling_mask).
        """
        toks = self.tokenizer.batch_encode_plus(
            transformed_chunks,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        hidden_states = self.model(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
        )[0]

        pooling_mask = toks["attention_mask"].clone()
        return hidden_states, pooling_mask

    def embed_sixtrack(
        self,
        sequences: list[str],
        cds: list[np.ndarray],
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Batch embed sequences using mRNABERT with 6-track input.

        Expects binary encoded tracks denoting the beginning of each codon
        in the CDS.

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks for sequences.
            splice: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (1, 768)
        """
        _ = splice  # Unused

        all_chunks = []
        chunk_counts = []

        for seq, c in zip(sequences, cds):
            chunks = self.chunk_sequence_cds_aware(seq, c, self.max_length - 2)
            for chunk_seq, chunk_cds in chunks:
                transformed = self.separate_utr_cds(chunk_seq, chunk_cds)
                all_chunks.append(transformed[0])
            chunk_counts.append(len(chunks))

        hidden_states, pooling_mask = self._forward_chunks_sixtrack(all_chunks)

        seq_embeddings = []
        chunk_ptr = 0

        for num_chunks in chunk_counts:
            seq_hidden = hidden_states[chunk_ptr:chunk_ptr + num_chunks]
            seq_mask = pooling_mask[chunk_ptr:chunk_ptr + num_chunks]

            hidden = seq_hidden.reshape(-1, seq_hidden.shape[-1])
            mask = seq_mask.reshape(-1).bool()

            masked_hidden = hidden[mask]
            seq_embeddings.append(agg_fn(masked_hidden))

            chunk_ptr += num_chunks

        return seq_embeddings

    def embed_sequence_sixtrack(
        self,
        sequence: str,
        cds: np.ndarray,
        splice: np.ndarray,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Embed single sequence using mRNABERT with 6-track input.

        Legacy single-sequence wrapper for embed_sixtrack.

        Args:
            sequence: Sequence to embed.
            cds: CDS track for sequence to embed.
            splice: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Tensor representing embedded sequence.
        """
        return self.embed_sixtrack(
            [sequence],
            [cds],
            [splice],
            agg_fn
        )[0]

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
