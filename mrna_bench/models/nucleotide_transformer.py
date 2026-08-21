from collections.abc import Callable
import warnings

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
)


class NucleotideTransformer(EmbeddingModel):
    """Inference wrapper for NucleotideTransformer.

    NucleotideTransformer is a transformer based DNA foundation model
    pre-trained using MLM on a variety of pre-training datasets ranging from
    the 1000 (human) genomes project to a multi-species dataset, across
    a wide range of parameters. Input is tokenized to 6-mers where possible,
    with a maximum tokenized sequence length of 1000.

    Link: https://github.com/instadeepai/nucleotide-transformer
    """

    default_version = "2.5b-multi-species"
    valid_versions = [
        "2.5b-multi-species",
        "2.5b-1000g",
        "500m-human-ref",
        "500m-1000g",
        "v2-50m-multi-species",
        "v2-100m-multi-species",
        "v2-250m-multi-species",
        "v2-500m-multi-species",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2"
    ]
    hookable_layer_patterns = [r"(?:esm\.)?encoder\.layer\.\d+"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "nt-" + model_version

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize NucleotideTransformer inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions are: {
                "2.5b-multi-species",
                "2.5b-1000g",
                "500m-human-ref",
                "500m-1000g",
                "v2-50m-multi-species",
                "v2-100m-multi-species",
                "v2-250m-multi-species",
                "v2-500m-multi-species"
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
            from transformers import AutoTokenizer, AutoModelForMaskedLM
        except ImportError:
            raise ImportError((
                "Install base_models optional dependency to use "
                "NucleotideTransformer."
            ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        # NTv2 do not support sdpa or FA2
        if "v2" in model_version and attn_implementation != "eager":
            warnings.warn(
                "NucleotideTransformer v2 models only support eager attention."
                " Defaulting to eager attention implementation."
            )

            self.attn_implementation = "eager"

        dtype = self._get_inference_dtype()

        language_model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(self.device)
        self._set_logits_model(language_model)

        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = (self.max_length - 2) * 6

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask
            excludes padding and CLS. NT has no EOS/SEP token.
        """
        toks = self.tokenizer.batch_encode_plus(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        attention_mask = toks["input_ids"] != self.tokenizer.pad_token_id

        torch_outs = self.model(
            toks["input_ids"],
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = torch_outs["hidden_states"][-1]

        # NT has no EOS/SEP token — only CLS at position 0 is excluded.
        pooling_mask = toks["attention_mask"].clone()
        pooling_mask[:, 0] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using NucleotideTransformer.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (hidden_dim,)
        """
        _, _ = cds, splice

        # NT uses 6-mer tokenization. max_length is in tokens.
        # Each token covers ~6 nucleotides, so max nucleotides is approx
        # (max_length - 2) * 6
        max_chunk_length = (self.max_length - 2) * 6

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=max_chunk_length,
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
        """Extract per-layer representations from NucleotideTransformer.

        Uses 6-mer tokenization; each token covers ~6 nucleotides.

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
        max_chunk_length = (self.max_length - 2) * 6

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            toks = self.tokenizer.batch_encode_plus(
                seqs,
                return_tensors="pt",
                padding=False,
            )
            toks["attention_mask"] = (
                toks["input_ids"] != self.tokenizer.pad_token_id
            ).long()
            return toks.to(self.device)  # type: ignore[return-value]

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=max_chunk_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
