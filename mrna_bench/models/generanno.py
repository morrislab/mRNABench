from collections.abc import Callable

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class GENERanno(EmbeddingModel):
    """Inference wrapper for GENERanno.

    GENERanno is a Transformer-encoder genomic foundation model designed for
    metagenomic annotation. It operates at single-nucleotide resolution with
    bidirectional attention over sequences up to 8k base pairs.

    The base checkpoints are pretrained on large-scale DNA corpora (e.g.,
    715B bp prokaryotic and 386B bp eukaryotic variants). Specialized
    "cds-annotator" checkpoints are finetuned for metagenomic CDS calling.

    Link: https://github.com/GenerTeam/GENERanno
    """

    default_version = "eukaryote-0.5b-base"
    valid_versions = [
        "prokaryote-0.5b-base",
        "prokaryote-0.5b-cds-annotator",
        "eukaryote-0.5b-base",
        "eukaryote-1.2b-cds-annotator-preview",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})
    # GENERanno overflows in FP16 under FA; need BF16 for range
    flash_attn_dtype = torch.bfloat16
    _BASE_VERSIONS = {
        "prokaryote-0.5b-base",
        "eukaryote-0.5b-base",
    }

    @classmethod
    def behaviors_for_version(
        cls,
        model_version: str,
    ) -> frozenset[ModelBehavior]:
        """Return MLM behavior only for base checkpoints.

        Args:
            model_version: Version whose supported behaviors to retrieve.

        Returns:
            Embedding behavior, plus pseudo-likelihood for base checkpoints.
        """
        behaviors = super().behaviors_for_version(model_version)
        if model_version in cls._BASE_VERSIONS:
            return behaviors | {ModelBehavior.PSEUDO_LIKELIHOOD}
        return behaviors

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "generanno-" + model_version

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize GENERanno inference wrapper.

        Args:
            model_version: Version of GENERanno to load. Valid values are: {
                "prokaryote-0.5b-base",
                "prokaryote-0.5b-cds-annotator",
                "eukaryote-0.5b-base",
                "eukaryote-1.2b-cds-annotator-preview",
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
            from transformers import (
                AutoModel,
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use GENERanno."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "GenerTeam/GENERanno-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        dtype = self._get_inference_dtype()

        model_class = (
            AutoModelForMaskedLM
            if model_version in self._BASE_VERSIONS else AutoModel
        )
        loaded_model = model_class.from_pretrained(
            "GenerTeam/GENERanno-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(self.device)
        if model_version in self._BASE_VERSIONS:
            self._set_logits_model(loaded_model)
        else:
            self.model = loaded_model

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        # Set pad_token to eos_token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = 8192  # based on technical report
        self.sequence_score_chunk_length = self.max_length - 2

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
            chunks,
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
        """Embed sequences using GENERanno.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1280,) for 0.5b models
            - default (mean): (2048,) for 1.2b models
        """
        _, _ = cds, splice

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
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
        """Extract per-layer representations from GENERanno.

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
                seqs,
                add_special_tokens=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length - 2,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
