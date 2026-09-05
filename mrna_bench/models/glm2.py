from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior
from mrna_bench.utils import get_model_weights_path


class GLM2(EmbeddingModel):
    """Inference wrapper for gLM2.

    gLM2 is a transformer-based genomic language model pretrained on
    protein and DNA sequences from prokaryotic operons, using MLM with
    a mixed nucleotide-amino acid vocabulary. DNA inputs are lowercased
    and prefixed with a strand marker by the tokenizer.

    Link: https://github.com/TattaBio/gLM2
    """

    default_version = "gLM-150M"
    valid_versions = ["gLM-150M", "gLM-650M"]
    default_attn_implementation = "sdpa"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layers\.\d+"]
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
        """Initialize gLM2.

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
                "to use gLM2."
            )

        hub_id = "Taykhoom/{}".format(model_version)
        cache_dir = get_model_weights_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            auto_prepare_dna=True,
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
        # One strand-marker token is prepended to every chunk.
        self.sequence_score_chunk_length = self.max_length - 1

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        _ = cds, splice
        return self.tokenizer(  # type: ignore[no-any-return]
            sequence,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

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
        special_ids = torch.tensor(
            self.tokenizer.all_special_ids,
            device=self.device,
        )
        special = torch.isin(tokens["input_ids"], special_ids)
        pooling_mask = tokens["attention_mask"].bool() & ~special
        return hidden, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool,
    ) -> list[torch.Tensor]:
        """Embed sequences using gLM2."""
        _, _ = cds, splice
        return self._embed_with_chunking(
            sequences,
            self.max_length - 1,
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
        """Extract per-layer representations from gLM2."""
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
            max_chunk_length=self.max_length - 1,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
