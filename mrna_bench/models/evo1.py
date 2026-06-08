from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class Evo1(EmbeddingModel):
    """Inference wrapper for Evo1.

    Evo1 is a StripedHyena-based DNA foundation model trained on the
    OpenGenome dataset using an autoregressive scheme at single nucleotide,
    byte level resolution. Owing to its StripedHyena backbone, it has a near
    linear scaling of compute and memory relative to its context window.

    Note: StripedHyena's convolutions don't fully isolate sequences in batched
    mode even with padding mask. This implementation uses single-sequence
    processing for consistency.

    Link: https://github.com/evo-design/evo
    """

    default_version = "Evo1-1.5-7B-8K"
    valid_versions = [
        "Evo1-1.5-7B-8K",
        "Evo1-1-7B-8K",
        "Evo1-1-7B-131K",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]

    # Attention QKV+output and MLP gates/projections in StripedHyena blocks.
    # These names are preserved by the HuggingFace port, but should be verified
    # against each remote-code revision before LoRA fine-tuning.
    lora_target_modules = [
        "Wqkv", "out_proj",
        "l1", "l2", "l3",
    ]

    version_to_num_layers = {
        "Evo1-1.5-7B-8K": 32,
        "Evo1-1-7B-8K": 32,
        "Evo1-1-7B-131K": 32,
    }

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Retrieve shortened name for model version.

        Args:
            model_version: Version of model to fetch short name for.

        Returns:
            Shortened name of model version.
        """
        short_names = {
            "Evo1-1-7B-8K": "evo-1-8k-base",
            "Evo1-1-7B-131K": "evo-1-131k-base",
            "Evo1-1.5-7B-8K": "evo-1.5-8k-base",
        }
        return short_names.get(model_version, model_version.replace("_", "-"))

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Evo1.

        Args:
            model_version: HuggingFace repo basename under ``Taykhoom/``.
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        effective_attn = (
            attn_implementation or self.default_attn_implementation
        )
        super().__init__(model_version, device, effective_attn)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use Evo1."
            )

        hub_id = f"Taykhoom/{model_version}"
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )
        self.model = AutoModel.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
        ).to(self.device)
        self.max_length = self.tokenizer.model_max_length

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered Evo1 representation labels exposed by hidden_states."""
        num_layers = self.version_to_num_layers[self.model_version]
        return [f"blocks.{idx}" for idx in range(num_layers)] + ["norm"]

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> torch.Tensor:
        """Embed a single sequence using Evo1.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Tensor representing embedded sequence with leading batch dimension.
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        chunk_embeddings = []
        for chunk in chunks:
            toks = self.tokenizer([chunk], return_tensors="pt").to(self.device)
            outputs = self.model(**toks, output_hidden_states=True)
            chunk_embeddings.append(outputs.last_hidden_state[0])

        return agg_fn(torch.cat(chunk_embeddings, dim=0)).float().unsqueeze(0)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> list[torch.Tensor]:
        """Embed sequences using mean-pooled final HuggingFace hidden states.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            List of embeddings with item shape depending on agg_fn.
        """
        _, _ = cds, splice
        return [
            self.embed_sequence(sequence, agg_fn=agg_fn).squeeze(0)
            for sequence in sequences
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
        """Extract per-layer representations from Evo1.

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
                return_tensors="pt",
                padding=False,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
