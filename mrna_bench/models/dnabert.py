from collections.abc import Callable

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class DNABERT2(EmbeddingModel):
    """Inference wrapper for DNA-BERT2.

    DNABERT2 is a transformer-based DNA foundation model that uses BPE and
    rotary positional encoding among other modern transformer improvements
    to allow for efficient inference. DNABERT2 is pre-trained using MLM
    on multi-species genomic dataset.

    Link: https://github.com/MAGICS-LAB/DNABERT_2
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize DNABERT2 inference wrapper.

        Args:
            model_version: Version of model used; must be "dnabert2".
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        if model_version != "dnabert2":
            raise ValueError("Only dnabert2 model version available.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        config = BertConfig.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
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
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using DNABERT2.

        Args:
            sequence: Sequence to embed.
            overlap: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            DNABERT2 representation of sequence with shape (1 x 768).
        """
        if overlap != 0:
            raise ValueError("DNABERT2 does not chunk sequence.")

        inputs = self.tokenizer(sequence, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        hidden_states = self.model(inputs)[0]

        embedding_mean = agg_fn(hidden_states, dim=1)
        return embedding_mean

    def embed_sequence_sixtrack(self, sequence, cds, splice, overlap, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for DNABERT.")
