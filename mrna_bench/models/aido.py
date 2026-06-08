from collections.abc import Callable
from functools import partial

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


class AIDORNA(EmbeddingModel):
    """Inference wrapper for AIDO.RNA.

    AIDO.RNA is a transformer-based RNA foundation model. It is trained using
    masked language modelling on 42 million non-coding RNA sequences, with
    domain adaptation models available for protein coding sequences.

    Link: https://github.com/genbio-ai/ModelGenerator
    """

    default_version = "AIDO.RNA-650M-CDS"
    valid_versions = [
        "AIDO.RNA-650M",
        "AIDO.RNA-650M-CDS",
        "AIDO.RNA-1.6B",
        "AIDO.RNA-1.6B-CDS",
        "AIDO.RNA-1M-MARS",
        "AIDO.RNA-25M-MARS",
        "AIDO.RNA-300M-MARS",
    ]
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
            "AIDO.RNA-650M": "aido-rna-650m",
            "AIDO.RNA-650M-CDS": "aido-rna-650m-cds",
            "AIDO.RNA-1.6B": "aido-rna-1b600m",
            "AIDO.RNA-1.6B-CDS": "aido-rna-1b600m-cds",
            "AIDO.RNA-1M-MARS": "aido-rna-mars-1m",
            "AIDO.RNA-25M-MARS": "aido-rna-mars-25m",
            "AIDO.RNA-300M-MARS": "aido-rna-mars-300m",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize AIDO.RNA.

        Args:
            model_version: Version of model used. Valid versions: {
                "AIDO.RNA-650M",
                "AIDO.RNA-650M-CDS",
                "AIDO.RNA-1.6B",
                "AIDO.RNA-1.6B-CDS",
                "AIDO.RNA-1M-MARS",
                "AIDO.RNA-25M-MARS",
                "AIDO.RNA-300M-MARS",
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
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use AIDO.RNA."
            )

        hub_id = "Taykhoom/{}".format(model_version)

        self.tokenizer = AutoTokenizer.from_pretrained(
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
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(device)
        self.max_length = self.tokenizer.model_max_length

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and special tokens (CLS/SEP).
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state
        pooling_mask = toks["attention_mask"].clone()

        # Exclude special tokens (CLS at pos 0, SEP at last real pos)
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using AIDO.RNA.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1280,) for 650M model
            - default (mean): (2048,) for 1.6B model
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
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
        """Extract per-layer representations from AIDO.RNA.

        Args:
            sequences: RNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            return self.tokenizer(  # type: ignore[return-value]
                seqs,
                return_tensors="pt",
                add_special_tokens=True,
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
