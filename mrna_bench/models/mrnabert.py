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
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "mRNABERT": "mrnabert",
        }
        return short_name_map[model_version]

    # the mlps have different target modules for LoRA
    lora_target_modules = ["Wqkv", "dense", "gated_layers", "wo"]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize mRNABERT inference wrapper.

        Args:
            model_version: Version of model to load.
                Only "mRNABERT" is supported.
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use mRNABERT."
            )

        hub_id = "Taykhoom/{}".format(model_version)

        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.config = AutoConfig.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        self.config.attn_implementation = attn_implementation

        self.model = AutoModel.from_pretrained(
            hub_id,
            trust_remote_code=True,
            add_pooling_layer=False,
            cache_dir=get_model_weights_path(),
            config=self.config,
        ).to(self.device)
        # ALiBi allows arbitrary lengths; model_max_length (from config) caps
        # the chunk size to avoid quadratic-memory OOM.
        self.max_length = self.tokenizer.model_max_length

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
             - default (mean): (768,)
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
            return_special_tokens_mask=True,
        ).to(self.device)

        special_tokens_mask = toks["special_tokens_mask"]
        attention_mask = 1 - special_tokens_mask

        hidden_states = self.model(
            input_ids=toks["input_ids"],
            attention_mask=attention_mask,
        ).last_hidden_state

        pooling_mask = attention_mask

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
             - default (mean): (768,)
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
    ) -> list[tuple[list[str], np.ndarray]]:
        """Chunk sequence while respecting codon boundaries.

        Args:
            sequence: Full RNA sequence.
            cds: CDS track for sequence (one entry per nucleotide).
            chunk_length: Maximum number of tokens per chunk.
        Returns:
            List of (token list chunk, cds chunk) tuples.
        """
        tokens = list(sequence)
        n = len(tokens)

        starts = np.where(cds != 0)[0]
        codon_starts = set(starts.tolist())

        if not codon_starts:
            return [
                (tokens[i:i + chunk_length], cds[i:i + chunk_length])
                for i in range(0, n, chunk_length)
            ]

        chunks = []
        i = 0

        while i < n:
            end = min(i + chunk_length, n)

            while end > i and any((end - k) in codon_starts for k in (1, 2)):
                end -= 1

            # if we couldn't find a codon boundary, just break at chunk_length
            if end == i:
                end = min(i + chunk_length, n)

            chunks.append((tokens[i:end], cds[i:end]))
            i = end

        return chunks

    def separate_utr_cds(
        self,
        tokens: list[str],
        cds: np.ndarray,
    ) -> list[str]:
        """Add spacing to separate UTR and CDS regions.

        mRNABERT requires that the UTRs have single character separation
        and the CDS regions have 3 character separation.

        I.E. for a sequence with UTR-CDS-UTR structure with the following
        sequence: "AACTGCGTG" and CDS track: [0,0,1,0,0,0,0,0,0],
        the returned sequence would be: "A A CTG C G T G"

        Args:
            tokens: Per-nucleotide token list for the sequence chunk (from
                chunk_sequence_cds_aware).
            cds: CDS track matching the token list length.
        Returns:
            List containing a single formatted string for the tokenizer.
        """
        starts = np.where(cds != 0)[0]
        if len(starts) == 0:
            return [" ".join(tokens)]

        start = int(starts[0])
        end = min(int(starts[-1]) + 3, len(tokens))

        parts = []

        if start > 0:
            parts.append(" ".join(tokens[:start]))

        # CDS: group tokens into 3-nucleotide codons. Truncated codons at the
        # end are kept as-is (not extended or padded).
        cds_items = []
        for j in range(start, end, 3):
            cds_items.append(''.join(tokens[j:j + 3]))
        parts.append(" ".join(cds_items))

        if end < len(tokens):
            parts.append(" ".join(tokens[end:]))

        return [" ".join(parts)]

    def extract(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        layers: list[int | str] | None = None,
        return_attentions: bool = False,
        offload_to_cpu: bool = True,
    ) -> tuple[
        dict[str, list[list[torch.Tensor]]],
        dict[str, list[list[torch.Tensor]] | None],
    ]:
        """Extract per-layer representations from mRNABERT.

        Requires CDS tracks for the 6-track tokenization scheme.

        Args:
            sequences: RNA sequences.
            cds: CDS tracks (required for 6-track encoding).
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        if cds is None:
            raise ValueError(
                "CDS tracks must be provided for mRNABERT extract()."
            )
        _ = splice

        def _chunk(seq_cds: tuple) -> list[tuple]:
            return self.chunk_sequence_cds_aware(
                seq_cds[0], seq_cds[1], self.max_length - 2
            )

        def _tokenize(items: list[tuple]) -> dict[str, torch.Tensor]:
            chunk_seq, chunk_cds = items[0]
            transformed = self.separate_utr_cds(chunk_seq, chunk_cds)
            raw = self.tokenizer.batch_encode_plus(
                transformed,
                add_special_tokens=True,
                padding=False,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            return {
                "input_ids": raw["input_ids"].to(self.device),
                "attention_mask": (
                    1 - raw["special_tokens_mask"]
                ).to(self.device),
            }

        return self._standard_hf_extract(
            sequences=list(zip(sequences, cds)),
            tokenize_fn=_tokenize,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
            chunk_fn=_chunk,
        )
