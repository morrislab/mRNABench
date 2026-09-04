from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class UTRBERT(EmbeddingModel):
    """Inference wrapper for 3UTRBERT.

    3UTRBERT is a transformer-based mRNA foundation model pretrained on the
    3'UTR regions of 100k RNA sequences from gencode using MLM. Various
    versions of 3UTRBERT are available with different k-mer sizes (3, 4, 5, 6).

    Link: https://github.com/yangyn533/3UTRBERT
    """

    default_version = "UTRBERT-6mer"
    valid_versions = [
        "UTRBERT-3mer",
        "UTRBERT-4mer",
        "UTRBERT-5mer",
        "UTRBERT-6mer",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]
    uses_rna_alphabet = True
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "UTRBERT-3mer": "utrbert-3mer",
            "UTRBERT-4mer": "utrbert-4mer",
            "UTRBERT-5mer": "utrbert-5mer",
            "UTRBERT-6mer": "utrbert-6mer",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize 3UTRBERT.

        Args:
            model_version: Version of model to load. Valid versions: {
                "UTRBERT-3mer", "UTRBERT-4mer", "UTRBERT-5mer",
                "UTRBERT-6mer"
            }
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use 3UTRBERT."
            )

        hub_id = "Taykhoom/{}".format(model_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        dtype = self._get_inference_dtype()
        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)
        self.k = self.tokenizer.kmer
        self.max_length = self.tokenizer.model_max_length
        self.max_kmer_tokens = self.max_length - 2
        self.max_chunk_length = self.max_kmer_tokens + self.k - 1
        self.sequence_score_chunk_length = self.max_chunk_length

    def _chunk_sequence_for_kmers(self, sequence: str) -> list[str]:
        if len(sequence) < self.k:
            raise ValueError(
                f"UTRBERT-{self.k}mer requires at least {self.k} nucleotides."
            )
        total_kmers = len(sequence) - self.k + 1
        return [
            sequence[start:start + self.max_chunk_length]
            for start in range(0, total_kmers, self.max_kmer_tokens)
        ]

    def _score_chunks(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray | None, np.ndarray | None]]:
        sequence = sequence.replace("T", "U")
        chunks = self._chunk_sequence_for_kmers(sequence)
        starts = range(
            0,
            len(sequence) - self.k + 1,
            self.max_kmer_tokens,
        )
        return [
            (
                chunk,
                None if cds is None else cds[start:start + len(chunk)],
                None if splice is None else splice[start:start + len(chunk)],
            )
            for start, chunk in zip(starts, chunks)
        ]

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask) tensors.
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state

        pooling_mask = toks["attention_mask"].clone()
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequence using only 3'UTR region using 3UTRBERT.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        all_chunks = []
        chunk_counts = []
        for sequence in sequences:
            chunks = self._chunk_sequence_for_kmers(sequence)
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        hidden_states, pooling_mask = self._forward_chunks(all_chunks)
        embeddings = []
        chunk_start = 0
        for chunk_count in chunk_counts:
            chunk_end = chunk_start + chunk_count
            hidden = hidden_states[chunk_start:chunk_end].reshape(
                -1,
                hidden_states.shape[-1],
            )
            mask = pooling_mask[chunk_start:chunk_end].reshape(-1).bool()
            embeddings.append(agg_fn(hidden[mask]))
            chunk_start = chunk_end
        return embeddings

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
        """Extract per-layer representations from 3UTRBERT.

        Args:
            sequences: RNA sequences (T or U bases; T->U applied internally).
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            return self.tokenizer(  # type: ignore[return-value]
                seqs, return_tensors="pt", padding=False
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
            chunk_fn=self._chunk_sequence_for_kmers,
        )
