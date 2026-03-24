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

    default_version = "dnabert2"
    valid_versions = ["dnabert2"]

    def __init__(self, model_version: str, device: torch.device):
        """Initialize DNABERT2 inference wrapper.

        Args:
            model_version: Version of model used; must be "dnabert2".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
            from transformers.models.bert.configuration_bert import BertConfig
            from transformers.models.bert.modeling_bert import BertModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use DNABERT2."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "czl/dnabert2",
            trust_remote_code=True,
            clean_up_tokenization_spaces=True,
            cache_dir=get_model_weights_path()
        )

        self.config = BertConfig.from_pretrained(
            "czl/dnabert2",
            cache_dir=get_model_weights_path(),
            add_pooling_layer=False
        )

        self.model = AutoModel.from_pretrained(
            "czl/dnabert2",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            config=self.config,
        ).to(self.device)

        # Reset AutoModel mapping to use default BertConfig for scenarios
        # where additional non-DNABERT loading occurs.
        AutoModel._model_mapping.register(
            BertConfig,
            (BertModel, BertModel),
            exist_ok=True
        )

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
            - default (mean): (1, 768)
        """
        _, _ = cds, splice

        toks = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks)[0]

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
