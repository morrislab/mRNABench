from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


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

    max_length = 1026

    def __init__(self, model_version: str, device: torch.device):
        """Initialize PlantRNAFM inference wrapper.

        Args:
            model_version: Version of model to load.
                    Only "plant_rnafm" is supported.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use PlantRNAFM."
            )

        self.config = AutoConfig.from_pretrained(
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
        self.config.mask_token_id = self.tokenizer.mask_token_id

        self.model = AutoModel.from_pretrained(
            "yangheng/PlantRNA-FM",
            trust_remote_code=True,
            config=self.config,
            cache_dir=get_model_weights_path(),
        ).to(device)

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
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using PlantRNAFM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1, 480)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )

        # # PlantRNA-FM does not properly respect attention_mask, so padding
        # # from longer sequences bleeds into shorter ones. Process each
        # # sequence individually to avoid ragged-batch inconsistency.
        # results = []
        # for seq in sequences:
        #     results.extend(self._embed_with_chunking(
        #         sequences=[seq],
        #         max_chunk_length=self.max_length - 2,
        #         embed_fn=self._forward_chunks,
        #         agg_fn=agg_fn,
        #     ))
        # return results
