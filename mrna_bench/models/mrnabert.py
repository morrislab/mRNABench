from collections.abc import Callable

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


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
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

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
            from transformers import (
                AutoConfig,
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
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

        dtype = self._get_inference_dtype()
        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            config=self.config,
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(self.device)
        self._set_logits_model(language_model)
        # ALiBi allows arbitrary lengths; model_max_length (from config) caps
        # the chunk size to avoid quadratic-memory OOM.
        self.max_length = self.tokenizer.model_max_length

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize UTR nucleotides and CDS codons for masked scoring."""
        _ = splice
        if cds is None:
            raise ValueError("mRNABERT scoring requires cds tracks.")
        transformed = self.separate_utr_cds(list(sequence), cds)[0]
        return self.tokenizer(  # type: ignore[no-any-return]
            transformed,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    def _score_chunks(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> list[
        tuple[str, np.ndarray | None, np.ndarray | None]
    ]:
        """Chunk scoring inputs without splitting CDS codons."""
        if cds is None:
            raise ValueError("mRNABERT scoring requires cds tracks.")
        chunks = self.chunk_sequence_cds_aware(
            sequence, cds, self.max_length - 2
        )
        offset = 0
        score_chunks: list[
            tuple[str, np.ndarray | None, np.ndarray | None]
        ] = []
        for chunk_tokens, chunk_cds in chunks:
            chunk_length = len(chunk_tokens)
            score_chunks.append((
                "".join(chunk_tokens),
                chunk_cds,
                (
                    None if splice is None
                    else splice[offset:offset + chunk_length]
                ),
            ))
            offset += chunk_length
        return score_chunks

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Batch embed sequences using mRNABERT.

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks for sequences.
            splice: List of splice tracks for sequences. Unused for mRNABERT.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (768,)
        """
        _ = splice
        if cds is None:
            raise ValueError(
                "CDS tracks must be provided for mRNABERT embedding."
            )
        self._validate_score_tracks(sequences, cds, None)

        self._warn_batch_size_reproducibility(len(sequences))

        all_chunks = []
        chunk_counts = []
        for sequence, cds_track in zip(sequences, cds):
            chunks = self.chunk_sequence_cds_aware(
                sequence, cds_track, self.max_length - 2
            )
            for chunk_sequence, chunk_cds in chunks:
                all_chunks.append(
                    self.separate_utr_cds(chunk_sequence, chunk_cds)[0]
                )
            chunk_counts.append(len(chunks))

        hidden_states, pooling_mask = self._forward_chunks_cds(all_chunks)
        embeddings = []
        chunk_start = 0
        for chunk_count in chunk_counts:
            chunk_end = chunk_start + chunk_count
            hidden = hidden_states[chunk_start:chunk_end].reshape(
                -1, hidden_states.shape[-1]
            )
            mask = pooling_mask[chunk_start:chunk_end].reshape(-1).bool()
            embeddings.append(agg_fn(hidden[mask]))
            chunk_start = chunk_end
        return embeddings

    def _forward_chunks_cds(
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

        attention_mask = toks["attention_mask"]
        pooling_mask = attention_mask * (1 - toks["special_tokens_mask"])

        hidden_states = self.model(
            input_ids=toks["input_ids"],
            attention_mask=attention_mask,
        ).last_hidden_state

        return hidden_states, pooling_mask

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
        self._validate_score_tracks(sequences, cds, None)
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
            )
            return {
                "input_ids": raw["input_ids"].to(self.device),
                "attention_mask": raw["attention_mask"].to(self.device),
            }

        return self._standard_hf_extract(
            sequences=list(zip(sequences, cds)),
            tokenize_fn=_tokenize,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
            chunk_fn=_chunk,
        )
