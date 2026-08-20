from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class HyenaDNA(EmbeddingModel):
    """Inference wrapper for HyenaDNA.

    HyenaDNA is a Hyena-based DNA foundation model trained on the human
    reference genome using an autoregressive scheme at single nucleotide
    resolution. Owing to its state-space backbone, it has an ultra long
    context window.

    HyenaDNA is causal, so right-padded tokens do not affect earlier sequence
    representations. Similar-length chunks are batched and trimmed before
    aggregation.

    Link: https://github.com/HazyResearch/hyena-dna
    """

    default_version = "hyenadna-medium-450k-seqlen-hf"
    valid_versions = [
        "hyenadna-large-1m-seqlen-hf",
        "hyenadna-medium-450k-seqlen-hf",
        "hyenadna-medium-160k-seqlen-hf",
        "hyenadna-small-32k-seqlen-hf",
        "hyenadna-tiny-16k-seqlen-d128-hf",
    ]
    default_attn_implementation = None
    valid_attn_implementations = None
    hookable_layer_patterns = [r"backbone\.layers\.\d+"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.CAUSAL_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-seqlen", "").replace("-hf", "")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize HyenaDNA inference wrapper.

        Support for HyenaDNA 1k models is currently omitted.

        Args:
            model_version: Version of model used. Valid versions are: {
                "hyenadna-large-1m-seqlen-hf",
                "hyenadna-medium-450k-seqlen-hf",
                "hyenadna-medium-160k-seqlen-hf",
                "hyenadna-small-32k-seqlen-hf",
                "hyenadna-tiny-16k-seqlen-d128-hf"
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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use HyenaDNA."
            )

        checkpoint = "LongSafari/{}".format(model_version)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self._set_logits_model(model)
        self.max_length = self._get_max_length()
        self.sequence_score_chunk_length = self.max_length

    def _get_max_length(self) -> int:
        """Get maximum sequence length for model."""
        context = self.model_version.split("-")[2]
        if context[-1] == "k":
            return int(context[:-1]) * 1000
        elif context[-1] == "m":
            return int(context[:-1]) * 1000000
        else:
            raise ValueError(
                "Invalid context length in model version. "
                "Expected 'k' or 'm' suffix."
            )

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using right-padded, length-bucketed batches.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (hidden_dim,)
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
            hidden_states = self.model(
                toks["input_ids"],
                output_hidden_states=True,
            ).hidden_states[-1]
            lengths = toks["input_ids"].ne(
                self.tokenizer.pad_token_id
            ).sum(dim=1)

            for batch_idx, (sequence_idx, chunk_idx, _) in enumerate(records):
                # The tokenizer appends EOS; right padding follows it.
                token_length = int(lengths[batch_idx].item()) - 1
                chunks_by_sequence[sequence_idx].append(
                    (
                        chunk_idx,
                        hidden_states[batch_idx, :token_length],
                    )
                )

        return [
            agg_fn(torch.cat([
                chunk for _, chunk in sorted(chunks, key=lambda item: item[0])
            ], dim=0), dim=0).float()
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
        """Extract per-layer representations from HyenaDNA.

        Uses forward hooks. Scores are None for all layers (Hyena, no
        attention weights).

        Args:
            sequences: DNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Ignored (no attention scores for Hyena).
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); scores are all None.
        """
        _, _ = cds, splice

        resolved = self._resolve_layer_paths(layers)
        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }
        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: None for p in resolved
        }

        for seq in sequences:
            chunks = self.chunk_sequence(seq, self.max_length)
            seq_hidden: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved
            }

            for chunk in chunks:
                toks = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                ).to(self.device)

                handles, activations = self._register_hooks(resolved)
                try:
                    self.model(toks["input_ids"])
                finally:
                    self._remove_hooks(handles)

                for path in resolved:
                    h = activations[path][0]
                    if h.dim() == 3:
                        h = h[0]  # (T, D)
                    seq_hidden[path].append(h.cpu() if offload_to_cpu else h)

            for path in resolved:
                hidden_out[path].append(seq_hidden[path])

        return hidden_out, score_out
