from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.utils import get_model_weights_path


class RNABERT(EmbeddingModel):
    """Inference Wrapper for RNABERT.

    RNABERT is a transformer based RNA foundation model pre-trained using
    both a MLM and structural alignment learning objective. RNABERT is
    pre-trained on 80K ncRNA sequences.

    Link: https://github.com/mana438/RNABERT
    """

    default_version = "RNABERT"
    valid_versions = ["RNABERT"]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]
    uses_rna_alphabet = True
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "RNABERT": "rnabert",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize RNABERT inference wrapper.

        Args:
            model_version: Must be "RNABERT".
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNABERT."
            )

        hub_id = "Taykhoom/{}".format(model_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        dtype = self._get_inference_dtype()
        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length

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

        # RNABERT's tokenizer adds no CLS/EOS, so all non-pad positions are
        # real content. Exclude only genuine special tokens (robust if the
        # tokenizer is later updated to add CLS/EOS).
        pooling_mask = toks["attention_mask"].clone()
        special_ids = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device
        )
        is_special = torch.isin(toks["input_ids"], special_ids)
        pooling_mask[is_special] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using RNABERT.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (120,)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length,
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
        """Extract per-layer representations from RNABERT.

        Args:
            sequences: RNA sequences (T or U bases accepted; T->U internally).
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
            max_chunk_length=self.max_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
