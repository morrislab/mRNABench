from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class PlantRNAFM(EmbeddingModel):
    """Inference wrapper for PlantRNAFM.

    PlantRNAFM is a transformer-based RNA foundation model pretrained on
    25M RNA sequences from 1,124 plant species (1KP). Pretraining uses
    MLM (BERT-style), RNA secondary structure prediction (predicted by
    ViennaRNA), and RNA region annotation prediction (e.g., CDS, 5' UTR,
    3' UTR). All objectives are optimized with cross-entropy loss.

    Link: https://github.com/yangheng95/PlantRNA-FM
    """

    default_version = "plant_rnafm"
    valid_versions = ["plant_rnafm"]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2"
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
        return model_version.replace("_", "-")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize PlantRNAFM inference wrapper.

        Args:
            model_version: Version of model to load.
                    Only "plant_rnafm" is supported.
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )
        self.max_length = 1026

        try:
            from transformers import (
                AutoConfig,
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use PlantRNAFM."
            )

        config = AutoConfig.from_pretrained(
            "yangheng/PlantRNA-FM",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "yangheng/PlantRNA-FM",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        # PlantRNA-FM's config sets mask_token_id=9, which is for Uracil, not
        # the <mask> special token (id=23). EsmEmbeddings.forward() scales
        # embeddings depending on the number of masked tokens in the input, so
        # the token dropout scale factor varies with the number of U's in the
        # input, producing different embeddings for the same sequence depending
        # on whether you batch it or embed it alone.
        # See: https://github.com/facebookresearch/esm/issues/267 for more on
        # token dropout. See huggingface->transformers->modeling_esm.py L210
        # for more about how mask_token_id is used in ESM's forward pass.
        config.mask_token_id = self.tokenizer.mask_token_id

        dtype = self._get_inference_dtype()

        language_model = AutoModelForMaskedLM.from_pretrained(
            "yangheng/PlantRNA-FM",
            trust_remote_code=True,
            config=config,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)
        self.sequence_score_chunk_length = self.max_length - 2

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
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using PlantRNAFM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (480,)
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
        """Extract per-layer representations from PlantRNAFM.

        Args:
            sequences: RNA sequences (T or U bases; T->U applied internally).
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
