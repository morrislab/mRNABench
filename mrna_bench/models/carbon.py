from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class Carbon(EmbeddingModel):
    """Inference wrapper for the Carbon family of genomic models.

    Carbon is a family of decoder-only autoregressive DNA models with a
    Llama-style architecture (RoPE positional embeddings, ``LlamaForCausalLM``)
    trained with next-token prediction on eukaryotic genes, mature/spliced
    mRNA, and bacterial genomes. The models share a hybrid tokenizer that
    encodes DNA as non-overlapping 6-mers (each DNA token spans ~6 bp) and
    English text as Qwen3 BPE; DNA inputs must be wrapped in ``<dna>...</dna>``
    so the tokenizer routes them through the 6-mer vocabulary rather than the
    text BPE. Native context is 8,192 tokens (~49 kbp).

    Link: https://github.com/huggingface/carbon
    """

    default_version = "Carbon-3B"
    valid_versions = [
        "Carbon-500M",
        "Carbon-3B",
        "Carbon-8B",
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

    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.lower()

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Carbon inference wrapper.

        Args:
            model_version: Version of Carbon to load. Valid values are: {
                "Carbon-500M",
                "Carbon-3B",
                "Carbon-8B",
            }
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(model_version, device, attn_implementation)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use Carbon."
            )

        hub_id = "HuggingFaceBio/{}".format(model_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        dtype = self._get_inference_dtype()

        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        )
        self._set_logits_model(loaded_model.to(self.device))

        # DNA tag tokens delimiting the 6-mer region; excluded from pooling.
        self.dna_tag_ids = set(
            self.tokenizer.convert_tokens_to_ids(["<dna>", "</dna>"])
        )

        # Native context (in tokens) varies by model: 8,192 for Carbon-500M,
        # 32,768 for Carbon-3B/8B. Read it from the config rather
        # than hardcode.
        model: Any = self.model
        self.context_length: int = (
            model.config.max_position_embeddings
        )

        # Reserve the two <dna>/</dna> tokens, then convert the remaining token
        # budget to nucleotides using the tokenizer's k-mer size.
        self.max_length = (self.context_length - 2) * self.tokenizer.k
        self.sequence_score_chunk_length = self.max_length
        self.causal_score_context_length = self.tokenizer.k

    def _wrap_dna(self, chunks: list[str]) -> list[str]:
        """Wrap raw nucleotide chunks in <dna>...</dna> for 6-mer tokenization.

        Args:
            chunks: Raw nucleotide sequence chunks.

        Returns:
            Chunks delimited by the DNA tags expected by the tokenizer.
        """
        return ["<dna>{}</dna>".format(chunk) for chunk in chunks]

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize with Carbon's required DNA delimiters."""
        _ = cds, splice
        return self.tokenizer(  # type: ignore[no-any-return]
            self._wrap_dna([sequence]),
            add_special_tokens=False,
            return_tensors="pt",
        )

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of raw nucleotide chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and the <dna>/</dna> delimiter tokens.
        """
        toks = self.tokenizer(
            self._wrap_dna(chunks),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length,
        ).to(self.device)

        hidden_states = self.model(
            **toks,
            output_hidden_states=True
        ).hidden_states[-1].float()

        is_dna_tag = torch.zeros_like(toks["input_ids"], dtype=torch.bool)
        for tag_id in self.dna_tag_ids:
            is_dna_tag |= toks["input_ids"] == tag_id
        pooling_mask = toks["attention_mask"] * (~is_dna_tag)

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using Carbon.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn. Default (mean)
            produces (hidden_dim,): 1024 for 500M, 3072 for 3B, 4096 for 8B.
        """
        _, _ = cds, splice

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length,
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
        """Extract per-layer representations from Carbon.

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
                self._wrap_dna(seqs),
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.context_length,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
