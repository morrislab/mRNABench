from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench.models import EmbeddingModel
from mrna_bench.utils import get_model_weights_path


class ERNIERNA(EmbeddingModel):
    """Inference Wrapper for ERNIE-RNA.

    ERNIE-RNA is a transformer based RNA foundation model pre-trained using
    MLM on 20M ncRNA sequences. ERNIE-RNA uses a custom attention map bias
    based on structural AU / GC / GU pairs. The backbones of fine-tuned
    secondary-structure and MRL variants are also available.

    Link: https://github.com/Bruce-ywj/ERNIE-RNA
    """

    default_version = "ERNIE-RNA"
    valid_versions = [
        "ERNIE-RNA",
        "ERNIE-RNA-SS",
        "ERNIE-RNA-MRL",
    ]
    default_attn_implementation = "eager"
    valid_attn_implementations = [
        "eager",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "ERNIE-RNA": "ernierna",
            "ERNIE-RNA-SS": "ernierna-ss",
            "ERNIE-RNA-MRL": "ernierna-mrl",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize ERNIE-RNA inference wrapper.

        Args:
            model_version: Version of ERNIE-RNA to use. Valid versions are:
                {"ERNIE-RNA", "ERNIE-RNA-SS", "ERNIE-RNA-MRL"}.
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
        except ImportError as exc:
            raise ImportError(
                "Install base_models optional dependency to use ERNIE-RNA."
            ) from exc

        hub_id = "Taykhoom/{}".format(model_version)
        weights_path = get_model_weights_path()

        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=weights_path
        )

        self.model = AutoModel.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=weights_path,
            attn_implementation=self.attn_implementation,
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
        """Embed sequences using ERNIE-RNA.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,) for `ERNIE-RNA`
            - default (mean): (768,) for `ERNIE-RNA-SS`
            - default (mean): (768,) for `ERNIE-RNA-MRL`
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
        """Extract per-layer representations from ERNIE-RNA.

        Args:
            sequences: RNA sequences (T or U bases; T→U applied internally).
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
                seqs, return_tensors="pt", padding=False
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length - 2,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
