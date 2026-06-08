from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class HyenaDNA(EmbeddingModel):
    """Inference wrapper for HyenaDNA.

    HyenaDNA is a Hyena-based DNA foundation model trained on the human
    reference genome using an autoregressive scheme at single nucleotide
    resolution. Owing to its state-space backbone, it has an ultra long
    context window.

    Note: HyenaDNA uses convolution-based Hyena operators which cannot mask
    padding tokens like attention-based models. Therefore, batched inference
    with variable-length sequences produces different embeddings than
    single-sequence inference. This implementation uses single-sequence
    processing for consistency.

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
            from transformers import AutoModel, AutoTokenizer
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

        model = AutoModel.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.tokenizer = tokenizer
        self.model = model
        self.max_length = self._get_max_length()

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

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Embed a single sequence using HyenaDNA.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Tensor representing embedded sequence.
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer(
                chunk,
                return_tensors="pt",
            ).to(self.device)

            hidden_states = self.model(toks["input_ids"])[0]

            # Exclude EOS token at end (last position)
            seq_hidden = hidden_states[0, :-1, :]
            chunk_embedding = agg_fn(seq_hidden, dim=0)
            embedding_chunks.append(chunk_embedding)

        # Aggregate across chunks
        if len(embedding_chunks) == 1:
            return embedding_chunks[0].unsqueeze(0).float()

        all_chunks = torch.stack(embedding_chunks, dim=0)
        return agg_fn(all_chunks).unsqueeze(0).float()

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using HyenaDNA.

        Processes sequences one at a time due to HyenaDNA's architectural
        limitation with padding (convolutions cannot mask padding tokens).

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

        all_embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding.squeeze(0))

        return all_embeddings

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
