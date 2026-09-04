from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class DNABERT(EmbeddingModel):
    """Inference wrapper for original k-mer DNABERT models.

    DNABERT is a BERT-base DNA foundation model pretrained on the human
    reference genome using overlapping k-mer tokenization. Unlike DNABERT2,
    original DNABERT requires sequences to be split into overlapping k-mers
    separated by spaces before tokenization.

    Link: https://github.com/jerryji1993/DNABERT
    """

    default_version = "DNABERT-6mer"
    valid_versions = [
        "DNABERT-3mer",
        "DNABERT-4mer",
        "DNABERT-5mer",
        "DNABERT-6mer",
    ]
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
            "DNABERT-3mer": "dnabert-3mer",
            "DNABERT-4mer": "dnabert-4mer",
            "DNABERT-5mer": "dnabert-5mer",
            "DNABERT-6mer": "dnabert-6mer",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize original k-mer DNABERT.

        Args:
            model_version: Version of model to load. Valid versions: {
                "DNABERT-3mer",
                "DNABERT-4mer",
                "DNABERT-5mer",
                "DNABERT-6mer"
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
                "Install base_models optional dependency to use DNABERT."
            )

        self.k = int(model_version.split("-")[1].replace("mer", ""))

        hub_id = "Taykhoom/{}".format(model_version)
        dtype = self._get_inference_dtype()
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": get_model_weights_path(),
            "attn_implementation": self.attn_implementation,
            "dtype": dtype,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )
        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            **model_kwargs,
        ).to(self.device)
        self._set_logits_model(language_model)

        self.max_length = self.tokenizer.model_max_length
        self.max_kmer_tokens = self.max_length - 2
        self.max_chunk_length = self.max_kmer_tokens + self.k - 1
        self.sequence_score_chunk_length = self.max_chunk_length

    def _seq_to_kmers(self, seq: str) -> str:
        """Convert a DNA sequence to overlapping space-delimited k-mers."""
        if len(seq) < self.k:
            raise ValueError(
                f"DNABERT-{self.k}mer requires at least {self.k} nucleotides."
            )
        return " ".join(
            seq[i:i + self.k]
            for i in range(len(seq) - self.k + 1)
        )

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize overlapping k-mers for masked scoring."""
        _ = cds, splice
        return self.tokenizer(  # type: ignore[no-any-return]
            self._seq_to_kmers(sequence),
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

    def _chunk_sequence_for_kmers(self, sequence: str) -> list[str]:
        """Split sequence into chunks with at most 510 k-mer tokens each."""
        if len(sequence) < self.k:
            raise ValueError(
                f"DNABERT-{self.k}mer requires at least {self.k} nucleotides."
            )

        total_kmers = len(sequence) - self.k + 1
        chunks = []
        for start in range(0, total_kmers, self.max_kmer_tokens):
            end = start + self.max_chunk_length
            chunks.append(sequence[start:end])
        return chunks

    def _score_chunks(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray | None, np.ndarray | None]]:
        """Chunk without dropping k-mers that cross chunk boundaries."""
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
        kmer_chunks = [self._seq_to_kmers(chunk) for chunk in chunks]

        toks = self.tokenizer(
            kmer_chunks,
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
        """Embed sequences using original k-mer DNABERT.

        Args:
            sequences: List of DNA sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,)
        """
        _, _ = cds, splice

        all_chunks = []
        chunk_counts = []
        for seq in sequences:
            chunks = self._chunk_sequence_for_kmers(seq)
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        hidden_states, pooling_mask = self._forward_chunks(all_chunks)

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
        """Extract per-layer representations from original k-mer DNABERT.

        Args:
            sequences: DNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            kmer_seqs = [self._seq_to_kmers(seq) for seq in seqs]
            return self.tokenizer(  # type: ignore[return-value]
                kmer_seqs,
                return_tensors="pt",
                padding=False,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
            chunk_fn=self._chunk_sequence_for_kmers,
        )
