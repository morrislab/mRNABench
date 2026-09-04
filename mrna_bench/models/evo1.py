from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
)


class Evo1(EmbeddingModel):
    """Inference wrapper for Evo1.

    Evo1 is a StripedHyena-based DNA foundation model trained on the
    OpenGenome dataset using an autoregressive scheme at single nucleotide,
    byte level resolution. Owing to its StripedHyena backbone, it has a near
    linear scaling of compute and memory relative to its context window.

    Causal right-padding isolates real tokens from padding. Similar-length
    chunks are batched and trimmed before aggregation.

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
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.CAUSAL_LIKELIHOOD,
    })

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
            from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.tokenizer.padding_side = "right"
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
        )
        loaded_model.config.use_cache = False
        self._set_logits_model(loaded_model.to(self.device))
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length
        self.causal_score_context_length = 1

    def _prepare_sequence_for_scoring(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        return "\x00" + sequence, cds, splice

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered Evo1 representation labels exposed by hidden_states."""
        num_layers = self.version_to_num_layers[self.model_version]
        return [f"blocks.{idx}" for idx in range(num_layers)] + ["norm"]

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool,
    ) -> list[torch.Tensor]:
        """Embed right-padded sequence batches using final hidden states.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            List of embeddings with item shape depending on agg_fn.
        """
        _, _ = cds, splice
        if not sequences:
            return []

        # Hyena's FFT size follows the padded length, so keep similar lengths
        # together to limit batch-dependent numerical drift.
        buckets: dict[int, list[tuple[int, int, str]]] = {}
        for sequence_idx, sequence in enumerate(sequences):
            for chunk_idx, chunk in enumerate(
                self.chunk_sequence(sequence, self.max_length)
            ):
                length_bucket = 1 << max(1, len(chunk)).bit_length()
                buckets.setdefault(length_bucket, []).append(
                    (sequence_idx, chunk_idx, chunk)
                )

        chunks_by_sequence: list[list[tuple[int, torch.Tensor]]] = [
            [] for _ in sequences
        ]
        for records in buckets.values():
            toks = self.tokenizer(
                [chunk for _, _, chunk in records],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(
                **toks,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[-1]
            lengths = toks["attention_mask"].sum(dim=1)

            for batch_idx, (sequence_idx, chunk_idx, _) in enumerate(records):
                chunks_by_sequence[sequence_idx].append(
                    (
                        chunk_idx,
                        hidden_states[
                            batch_idx, :int(lengths[batch_idx].item())
                        ],
                    )
                )

        return [
            agg_fn(torch.cat([
                chunk for _, chunk in sorted(chunks, key=lambda item: item[0])
            ], dim=0)).float()
            for chunks in chunks_by_sequence
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
