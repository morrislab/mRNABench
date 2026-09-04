from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
)


class Evo2(EmbeddingModel):
    """Inference wrapper for Evo2.

    Evo2 is a StripedHyena2-based DNA foundation model trained on the
    OpenGenome2 dataset using an autoregressive scheme at single nucleotide
    resolution. Owing to its StripedHyena2 backbone, it has an ultra long
    context window. The `base` variants can handle sequences up to 8192
    nucleotides in length while the larger variants can handle sequences up
    to 1 million nucleotides in length. While it can in principle handle
    sequences longer than 1 MB, due to GPU memory constraints, we limit
    the maximum sequence length to 1,000,000 nucleotides. This can be
    increased if more GPU memory is available.

    Link: https://github.com/ArcInstitute/evo2
    """

    default_version = "Evo2-7B-8K"
    valid_versions = [
        "Evo2-1B-8K",
        "Evo2-7B-8K",
        "Evo2-7B-262K",
        "Evo2-7B-1M",
        "Evo2-20B-1M",
        "Evo2-40B-8K",
        "Evo2-40B-1M",
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

    # Attention QKV+output and MLP gates/projections in StripedHyena2 blocks.
    # These names are preserved by the HuggingFace port, but should be verified
    # against each remote-code revision before LoRA fine-tuning.
    lora_target_modules = [
        "Wqkv", "out_proj",
        "l1", "l2", "l3",
    ]

    version_to_num_layers = {
        "Evo2-1B-8K": 25,
        "Evo2-7B-8K": 32,
        "Evo2-7B-262K": 32,
        "Evo2-7B-1M": 32,
        "Evo2-20B-1M": 24,
        "Evo2-40B-8K": 50,
        "Evo2-40B-1M": 50,
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
            "Evo2-1B-8K": "evo2-1b-base",
            "Evo2-7B-8K": "evo2-7b-base",
            "Evo2-7B-262K": "evo2-7b-262k",
            "Evo2-7B-1M": "evo2-7b",
            "Evo2-20B-1M": "evo2-20b",
            "Evo2-40B-8K": "evo2-40b-base",
            "Evo2-40B-1M": "evo2-40b",
        }
        return short_names.get(model_version, model_version.replace("_", "-"))

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Evo2.

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
                "Install base_models optional_dependency to use Evo2."
            )

        hub_id = f"Taykhoom/{model_version}"
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )
        self.tokenizer.padding_side = "right"
        model_kwargs: dict[str, Any] = {}
        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            **model_kwargs,
        )
        loaded_model.config.use_cache = False
        if "device_map" in model_kwargs:
            self.device = loaded_model.get_input_embeddings().weight.device
        else:
            loaded_model = loaded_model.to(self.device)
        self._set_logits_model(loaded_model)
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length
        # Middle block = num_layers // 2 (per the Evo2 HF port READMEs). Its
        # pre-norm output is concatenated with the final-layer embedding.
        self.middle_layer_idx = self.version_to_num_layers[model_version] // 2
        self.middle_layer_path = (
            f"blocks.{self.middle_layer_idx}.pre_norm"
        )

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered Evo2 representation labels exposed by hidden_states."""
        num_layers = self.version_to_num_layers[self.model_version]
        return [f"blocks.{idx}" for idx in range(num_layers)] + ["norm"]

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool,
    ) -> list[torch.Tensor]:
        """Embed right-padded sequence batches as middle + final states.

        Each embedding is the token-aggregated concatenation of the middle
        block's pre-norm representation and the final-layer hidden state, with
        feature dimension ``2 * hidden_size``. Similar-length chunks share a
        model call to limit padded computation.

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

        batching_enabled = (
            getattr(self.model.config, "long_fir_threshold", None) is None
        )
        # Hyena's FFT size follows the padded length, so keep similar lengths
        # together to limit batch-dependent numerical drift.
        buckets: dict[object, list[tuple[int, int, str]]] = {}
        for sequence_idx, sequence in enumerate(sequences):
            for chunk_idx, chunk in enumerate(
                self.chunk_sequence(sequence, self.max_length)
            ):
                if batching_enabled:
                    bucket: object = 1 << max(1, len(chunk)).bit_length()
                else:
                    bucket = (sequence_idx, chunk_idx)
                buckets.setdefault(bucket, []).append(
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
            outputs, captured = self._run_with_layer_capture(
                [self.middle_layer_path],
                lambda: self.model(
                    **toks,
                    output_hidden_states=False,
                    use_cache=False,
                ),
                detach=False,
            )
            middle_outputs = captured[self.middle_layer_path]
            if len(middle_outputs) != 1:
                raise RuntimeError(
                    "Failed to capture Evo2 middle hidden state."
                )
            middle_hidden = middle_outputs[0]
            middle_hidden = middle_hidden.to(outputs.last_hidden_state.device)
            combined = torch.cat(
                [middle_hidden, outputs.last_hidden_state],
                dim=-1,
            )
            lengths = toks["attention_mask"].sum(dim=1)

            for batch_idx, (sequence_idx, chunk_idx, _) in enumerate(records):
                chunks_by_sequence[sequence_idx].append(
                    (
                        chunk_idx,
                        combined[batch_idx, :int(lengths[batch_idx].item())],
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
        """Extract per-layer representations from Evo2.

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
