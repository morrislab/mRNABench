from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


class HelixmRNA(EmbeddingModel):
    """Inference wrapper for Helix-mRNA.

    Helix-mRNA is a RNA foundation model trained using a Mamba2 and transformer
    hybrid backbone. Helix-mRNA is pre-trained on 26M mRNAs from diverse
    eukaryotic and viral species.

    CDS tracks are ignored: the released weights were not trained on the
    codon-start token 'E' that upstream recommends inserting.

    Link: https://github.com/helicalAI/helical
    """

    default_version = "helix-mrna"
    valid_versions = ["helix-mrna"]
    default_attn_implementation = "sdpa"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    final_layer_pattern = r"norm_f"
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.CAUSAL_LIKELIHOOD,
    })

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Helix-mRNA model.

        Args:
            model_version: Must be "helix-mrna".
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
            raise ImportError("Helix-mRNA missing required dependencies.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Taykhoom/Helix-mRNA",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        dtype = self._get_inference_dtype()

        model: Any = AutoModelForCausalLM.from_pretrained(
            "Taykhoom/Helix-mRNA",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        )
        self._set_logits_model(model.to(self.device))

        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length

    def _sequence_attention(
        self,
        tokens: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Attend to sequence tokens only, not the appended [SEP].

        Args:
            tokens: Output of tokenize().

        Returns:
            Tokens with special tokens removed from the attention mask.
        """
        tokens["attention_mask"] = self.pooling_mask(tokens).long()
        return tokens

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
        tokens = self._sequence_attention(self.tokenize(chunks))
        hidden_states = self.model(**tokens).last_hidden_state
        return hidden_states, tokens["attention_mask"]

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
    ) -> list[torch.Tensor]:
        """Batch embed sequences using Helix-mRNA.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (256,)
        """
        _ = cds, splice  # Unused

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 1,  # Account for SEP token
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
        """Extract per-layer representations from Helix-mRNA.

        Attention extraction requires the model to be loaded with
        attn_implementation="eager". Flash Attention 2 does not materialise
        the (H, T, T) attention matrix; scores will be None for all layers
        when flash_attention_2 is active. Only layers whose block type is
        "attention" (only layer 3 in the default "M+M*M+M+" config) can
        produce attention weights; Mamba and MLP layers yield None.

        Args:
            sequences: RNA sequences (T or U; the tokenizer reads T as U).
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
            scores[layer] is None for Mamba and MLP layers regardless of
            return_attentions.
        """
        _ = cds, splice
        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=lambda seqs: self._sequence_attention(
                self.tokenize(seqs)
            ),
            max_chunk_length=self.max_length - 1,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
