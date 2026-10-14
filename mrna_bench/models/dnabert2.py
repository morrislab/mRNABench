from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


class DNABERT2(EmbeddingModel):
    """Inference wrapper for DNA-BERT2.

    DNABERT2 is a transformer-based DNA foundation model that uses BPE and
    ALiBi positional encoding among other modern transformer improvements
    to allow for efficient inference. DNABERT2 is pre-trained using MLM
    on multi-species genomic dataset.

    Link: https://github.com/MAGICS-LAB/DNABERT_2
    """

    default_version = "DNABERT2"
    valid_versions = ["DNABERT2"]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]
    lora_target_modules = ["Wqkv", "dense", "gated_layers", "wo"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "DNABERT2": "dnabert2",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize DNABERT2 inference wrapper.

        Args:
            model_version: Version of model used; must be "DNABERT2".
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
                "Install base_models optional_dependency to use DNABERT2."
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
        self.sequence_score_chunk_length = self.max_length - 2

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
    ) -> list[torch.Tensor]:
        """Embed sequences using DNABERT2.

        ALiBi positional encoding allows for arbitrary sequence lengths.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,)
        """
        _, _ = cds, splice

        hidden_states, pooling_mask = self._forward_chunks(sequences)
        return [
            agg_fn(hidden[mask])
            for hidden, mask in zip(hidden_states, pooling_mask)
        ]

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
        """Extract per-layer representations from DNABERT2.

        DNABERT2 uses ALiBi positional encoding and supports arbitrary
        sequence lengths without chunking. The patched bert_layers.py returns
        a standard BaseModelOutputWithPooling, so hf_extract is used directly.
        Attention weights require eager or sdpa (not flash_attention_2).

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

        return self._standard_hf_extract(
            sequences=sequences,
            max_chunk_length=self.max_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
