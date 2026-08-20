from collections.abc import Callable
from typing import Any

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class GENERator(EmbeddingModel):
    """Inference wrapper for GENERator.

    GENERator is a Transformer-based autoregressive genomic foundation model
    using k-mer tokenization. The original model is trained with standard
    next-token prediction on gene-centric functional regions to enable
    long-context generative modeling of eukaryotic genomes.

    GENERator-v2 retains the same backbone and tokenization but introduces
    Factorized Nucleotide Supervision (FNS), which decomposes each k-mer
    prediction into nucleotide-level likelihoods, and Genome Compression
    Pretraining (GCP), which concatenates functional regions to densify
    biological signal and induce next-gene prediction. v2 supports contexts
    up to 98k base pairs and includes eukaryotic and prokaryotic variants.

    Link: https://github.com/GenerTeam/GENERator
    """

    default_version = "v2-eukaryote-3b-base"
    valid_versions = [
        "eukaryote-1.2b-base",
        "v2-eukaryote-1.2b-base",
        "v2-prokaryote-1.2b-base",
        "eukaryote-3b-base",
        "v2-eukaryote-3b-base",
        "v2-prokaryote-3b-base",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.CAUSAL_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "generator-" + model_version

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize GENERator inference wrapper.

        Args:
            model_version: Version of GENERator to load. Valid values are: {
                "eukaryote-1.2b-base",
                "eukaryote-3b-base",
                "v2-eukaryote-1.2b-base",
                "v2-eukaryote-3b-base",
                "v2-prokaryote-1.2b-base",
                "v2-prokaryote-3b-base",
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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use GENERator."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "GenerTeam/GENERator-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        dtype = self._get_inference_dtype()

        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            "GenerTeam/GENERator-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        )
        self._set_logits_model(loaded_model.to(self.device))

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        # Set pad_token to eos_token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = 98_304  # based on technical report

        # Chunk length must be a multiple of k (k-mer-aligned boundaries) and
        # reserve 2 *tokens* for <s>/</s> so a full chunk fits max_pos_embeds.
        self.k = self.tokenizer.k
        self.max_chunk_length = ((self.max_length // self.k) - 2) * self.k
        self.sequence_score_chunk_length = self.max_chunk_length
        self.causal_score_context_length = self.k

    def _pad_to_kmer(self, chunks: list[str]) -> list[str]:
        """Right-pad each chunk with 'A' so its length is a multiple of k.

        GENERator's tokenizer maps any trailing sub-k-mer remainder to a
        single uninformative <oov> token (and discards its bases). Padding the
        remainder up to a full k-mer with 'A' preserves the leading bases.

        Args:
            chunks: Raw nucleotide chunks.

        Returns:
            Chunks right-padded with 'A' to the next multiple of k.
        """
        padded = []
        for chunk in chunks:
            remainder = len(chunk) % self.k
            if remainder:
                chunk = chunk + "A" * (self.k - remainder)
            padded.append(chunk)
        return padded

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Pad trailing bases before tokenizing for likelihood scoring."""
        _ = cds, splice
        return self.tokenizer(  # type: ignore[no-any-return]
            self._pad_to_kmer([sequence])[0],
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and special tokens (CLS/SEP).
        """
        toks = self.tokenizer(
            self._pad_to_kmer(chunks),
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        ).to(self.device)

        special_tokens_mask = toks.pop("special_tokens_mask")

        hidden_states = self.model(
            **toks,
            output_hidden_states=True
        ).hidden_states[-1]

        pooling_mask = 1 - special_tokens_mask

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using GENERator.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (2048,) for 1.2b models
            - default (mean): (3072,) for 3b models
        """
        _, _ = cds, splice

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_chunk_length,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )

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
        """Extract per-layer representations from GENERator.

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
            return self.tokenizer(  # type: ignore[return-value]
                self._pad_to_kmer(seqs),
                add_special_tokens=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_chunk_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
