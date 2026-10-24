from collections.abc import Callable
import warnings

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
)
from mrna_bench.models.pooling import MeanPool


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
    final_layer_pattern = r"(?:esm\.)?encoder\.emb_layer_norm_after"
    lora_target_modules = ["query", "key", "value", "dense"]
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

    def chunk_sequence(
        self,
        sequence: str,
        chunk_length: int | None,
    ) -> list[str]:
        """Split a sequence into consecutive chunks that fit the model.

        The tokenizer reads 6-mers, but leftover bases at the end of a chunk
        and the bases around an N become one token each, so a chunk can
        exceed max_length tokens. Such chunks are split in two at a 6-mer
        boundary until they fit.

        Args:
            sequence: Sequence to chunk.
            chunk_length: Length of each chunk; the last may be shorter.
                None keeps the sequence whole.

        Returns:
            Sequence chunks.
        """
        chunks = []
        for chunk in super().chunk_sequence(sequence, chunk_length):
            chunks.extend(self._fit_chunk(chunk))
        return chunks

    def _fit_chunk(self, chunk: str) -> list[str]:
        """Split a chunk until each piece fits max_length tokens."""
        if len(self.tokenizer(chunk)["input_ids"]) <= self.max_length:
            return [chunk]
        cut = len(chunk) // 12 * 6
        return self._fit_chunk(chunk[:cut]) + self._fit_chunk(chunk[cut:])

    def tokenize(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize sequences, masking padding by its token id.

        Args:
            sequences: Nucleotide sequences, or chunks of them.
            cds: Unused.
            splice: Unused.
            add_special_tokens: Whether to add the CLS token.

        Returns:
            Tokenizer inputs on the model's device.
        """
        tokens = super().tokenize(sequences, cds, splice, add_special_tokens)
        tokens["attention_mask"] = (
            tokens["input_ids"] != self.tokenizer.pad_token_id
        ).long()
        return tokens

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask
            excludes padding and CLS.
        """
        tokens = self.tokenize(chunks)
        torch_outs = self.model(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            encoder_attention_mask=tokens["attention_mask"],
            output_hidden_states=True
        )
        hidden_states = torch_outs["hidden_states"][-1]
        return hidden_states, self.pooling_mask(tokens)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
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

        return self._standard_hf_extract(
            sequences=sequences,
            max_chunk_length=max_chunk_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
