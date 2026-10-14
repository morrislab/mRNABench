from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


class RiNALMo(EmbeddingModel):
    """Inference wrapper for RiNALMo.

    RiNALMo is a transformer-based RNA foundation model trained on 36M ncRNA
    sequences using MLM and other modern architectural improvements such as
    RoPE, SwiGLU activations, and Flash Attention.

    Link: https://github.com/lbcb-sci/RiNALMo
    """

    default_version = "RiNALMo-mega"
    valid_versions = [
        "RiNALMo-micro",
        "RiNALMo-mega",
        "RiNALMo-giga"
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    final_layer_pattern = r"final_layer_norm"
    # Fused attention QKV + output, and the SwiGLU feed-forward projections.
    lora_target_modules = [
        "qkv_proj", "out_proj",
        "linear", "linear_gate", "ffn_down",
    ]
    uses_rna_alphabet = True
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "RiNALMo-micro": "rinalmo-micro",
            "RiNALMo-mega": "rinalmo-mega",
            "RiNALMo-giga": "rinalmo-giga",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize RiNALMo inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions: {
                "RiNALMo-giga", "RiNALMo-mega", "RiNALMo-micro"
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
                "Install base_models optional dependency to use RiNALMo."
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
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length - 2

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
    ) -> list[torch.Tensor]:
        """Embed sequences using RiNALMo.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (hidden_dim,)
        """
        _, _ = cds, splice
        effective_max = self.max_length - 2

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=effective_max,
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
        """Extract per-layer representations from RiNALMo.

        RiNALMo nominally uses Flash Attention, but the HuggingFace
        implementation falls back to eager attention when
        output_attentions=True is requested, so attention weights are
        available for all 33 transformer layers.

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
        return self._standard_hf_extract(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
