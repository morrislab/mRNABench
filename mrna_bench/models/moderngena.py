from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior
from mrna_bench.utils import get_model_weights_path


class ModernGENA(EmbeddingModel):
    """Inference wrapper for ModernGENA.

    ModernGENA is a ModernBERT-based DNA foundation model that uses
    pre-normalization, RoPE, local/global attention patterns, and GeGLU
    activations. It is pretrained on multi-species genomes with BPE
    tokenization.

    Link: https://github.com/AIRI-Institute/GENA_LM
    """

    default_version = "ModernGENA-base"
    valid_versions = ["ModernGENA-base", "ModernGENA-large"]
    default_attn_implementation = "sdpa"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize ModernGENA.

        Args:
            model_version: Version of model to load.
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(model_version, device, attn_implementation)

        try:
            from transformers import (
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency "
                "to use ModernGENA."
            )

        hub_id = "Taykhoom/{}".format(model_version)
        cache_dir = get_model_weights_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        dtype = self._get_inference_dtype()
        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            cache_dir=cache_dir,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length - 2

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        hidden = self.model(**tokens).last_hidden_state
        pooling_mask = tokens["attention_mask"].bool()
        for row, token_ids in enumerate(tokens["input_ids"]):
            special = self.tokenizer.get_special_tokens_mask(
                token_ids.tolist(),
                already_has_special_tokens=True,
            )
            pooling_mask[row] &= ~torch.tensor(
                special,
                dtype=torch.bool,
                device=self.device,
            )
        return hidden, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool,
    ) -> list[torch.Tensor]:
        """Embed sequences using ModernGENA."""
        _, _ = cds, splice
        return self._embed_with_chunking(
            sequences,
            self.max_length - 2,
            self._forward_chunks,
            agg_fn,
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
        """Extract per-layer representations from ModernGENA."""
        _, _ = cds, splice

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            return self.tokenizer(  # type: ignore[return-value]
                seqs,
                return_tensors="pt",
                padding=False,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length - 2,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
