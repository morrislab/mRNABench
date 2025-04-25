from collections.abc import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class CodonBERT(EmbeddingModel):
    """Inference wrapper for CodonBERT.

    CodonBERT is a transformer-based RNA language model that is
    pretrained on more than 10 million mRNA sequences from mammals,
    bacteria, and human viruses using MLM. It is specifically trained
    on coding regions of mRNA sequences, and is designed for predicting
    mRNA-specific properties.

    Link: https://github.com/Sanofi-Public/CodonBERT
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize CodonBERT inference wrapper.

        Args:
            model_version: Version of model used; must be "codonbert".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use CodonBERT."
            )

        if model_version != "codonbert":
            raise ValueError("Only codonbert model version available.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "lhallee/CodonBERT",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "lhallee/CodonBERT",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        ).to(self.device)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using CodonBERT.

        Args:
            sequence: Sequence to embed.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            CodonBERT representation of sequence with shape (1 x 768).
        """
        inputs = self.tokenizer(sequence, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        hidden_states = self.model(inputs)[0]

        embedding_mean = agg_fn(hidden_states, dim=1)
        return embedding_mean

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for CodonBERT.")
