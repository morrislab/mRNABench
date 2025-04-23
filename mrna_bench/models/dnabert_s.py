from collections.abc import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class DNABERTS(EmbeddingModel):
    """Inference wrapper for DNA-BERT2.

    DNABERT-S is a transformer-based DNA foundation model that builds on
    DNABERTS to produce species-aware embeddings for genomic sequences.
    It is trained using a contrastive learning objective which encourages
    grouping of DNA sequences from the same species and discourages grouping
    of sequences from different species. DNABERT-S is trained using microbial
    genomic sequences from viruses, fungi, and bacteria.

    Link: https://github.com/MAGICS-LAB/DNABERT_S
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize DNABERTS inference wrapper.

        Args:
            model_version: Version of model used; must be "dnabert-s".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
            from transformers.models.bert.configuration_bert import BertConfig
            from transformers.models.bert.modeling_bert import BertModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use DNABERTS."
            )

        if model_version != "dnabert-s":
            raise ValueError("Only dnabert-s model version available.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-S",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        config = BertConfig.from_pretrained(
            "zhihan1996/DNABERT-S",
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-S",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            config=config
        ).to(self.device)

        # Reset AutoModel mapping to use default BertConfig for scenarios
        # where additional non-DNABERT loading occurs.
        AutoModel._model_mapping.register(BertConfig, BertModel, exist_ok=True)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using DNABERTS.

        Args:
            sequence: Sequence to embed.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            DNABERTS representation of sequence with shape (1 x 768).
        """
        inputs = self.tokenizer(sequence, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        hidden_states = self.model(inputs)[0]

        embedding_mean = agg_fn(hidden_states, dim=1)
        return embedding_mean

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for DNABERT.")
