from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


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
            from transformers import AutoTokenizer, AutoModel, AutoConfig
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

        dtype = (
            torch.bfloat16
            if self.attn_implementation == "flash_attention_2"
            else torch.float32
        )
        self.model = AutoModel.from_pretrained(
            hub_id,
            trust_remote_code=True,
            add_pooling_layer=False,
            cache_dir=get_model_weights_path(),
            config=self.config,
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(self.device)
        # ALiBi allows arbitrary lengths; model_max_length (from config) caps
        # the chunk size to avoid quadratic-memory OOM.
        self.max_length = self.tokenizer.model_max_length

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
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

        toks = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state

        # Build pooling mask excluding CLS (pos 0) and SEP (last real pos)
        pooling_mask = toks["attention_mask"].clone()
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        # Apply masked aggregation per sequence
        embeddings = []
        for i in range(hidden_states.size(0)):
            mask = pooling_mask[i].bool()
            masked_hidden = hidden_states[i][mask]
            embeddings.append(agg_fn(masked_hidden))

        return embeddings

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

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            return self.tokenizer(  # type: ignore[return-value]
                seqs, return_tensors="pt", padding=False
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
