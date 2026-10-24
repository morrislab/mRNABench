from collections.abc import Callable

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


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
        # ALiBi handles any length; set to chunk on smaller GPUs
        self.max_chunk_length: int | None = None
        self.max_length = self.tokenizer.model_max_length

    def tokenize(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize UTR nucleotides and CDS codons.

        Args:
            sequences: Nucleotide sequences, or chunks of them.
            cds: CDS tracks aligned to sequences (required).
            splice: Unused.
            add_special_tokens: Whether to add CLS and SEP.

        Returns:
            Tokenizer inputs on the model's device.
        """
        _ = splice
        if cds is None:
            raise ValueError("mRNABERT requires cds tracks.")
        return self.tokenizer(  # type: ignore[no-any-return]
            [
                self.separate_utr_cds(list(sequence), track)[0]
                for sequence, track in zip(sequences, cds)
            ],
            add_special_tokens=add_special_tokens,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

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
        agg_fn: Callable = MeanPool()
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

        return self._embed_with_chunking(
            sequences=list(zip(sequences, cds)),
            max_chunk_length=self.max_chunk_length,
            embed_fn=self._forward_cds_chunks,
            agg_fn=agg_fn,
            chunk_fn=self._chunk_cds,
        )

    def _chunk_cds(self, item: tuple) -> list[tuple]:
        """Split a (sequence, cds) item without cutting codons.

        Args:
            item: Sequence and its CDS track.

        Returns:
            (token list, cds) chunks; one chunk unless max_chunk_length is
            set.
        """
        sequence, cds = item
        if self.max_chunk_length is None:
            return [(list(sequence), cds)]
        return self.chunk_sequence_cds_aware(
            sequence, cds, self.max_chunk_length
        )

    def _forward_cds_chunks(
        self,
        chunks: list[tuple],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of (token list, cds) chunks."""
        tokens = self.tokenize(
            ["".join(chunk) for chunk, _ in chunks],
            [cds for _, cds in chunks],
        )
        hidden_states = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        ).last_hidden_state
        return hidden_states, self.pooling_mask(tokens)

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

        return self._standard_hf_extract(
            sequences=list(zip(sequences, cds)),
            tokenize_fn=lambda items: self.tokenize(
                ["".join(items[0][0])], [items[0][1]]
            ),
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
            chunk_fn=self._chunk_cds,
        )
