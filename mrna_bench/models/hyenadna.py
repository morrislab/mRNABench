from collections.abc import Callable

import torch
from transformers import AutoModel, AutoTokenizer

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class HyenaDNA(EmbeddingModel):
    """Inference wrapper for HyenaDNA.

    HyenaDNA is a Hyena-based DNA foundation model trained on the human
    reference genome using an autoregressive scheme at single nucleotide
    resolution. Owing to its state-space backbone, it has an ultra long
    context window.

    Link: https://github.com/HazyResearch/hyena-dna
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-seqlen", "").replace("-hf", "")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize HyenaDNA inference wrapper.

        Support for HyenaDNA 1k models is currently omitted.

        Args:
            model_version: Version of model used. Valid versions are: {
                "hyenadna-large-1m-seqlen-hf",
                "hyenadna-medium-450k-seqlen-hf",
                "hyenadna-medium-160k-seqlen-hf",
                "hyenadna-small-32k-seqlen-hf",
                "hyenadna-tiny-16k-seqlen-d128-hf"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        checkpoint = "LongSafari/{}".format(model_version)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        model = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.tokenizer = tokenizer
        self.model = model

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using HyenaDNA.

        Args:
            sequence: Sequence to embed.
            overlap: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            HyenaDNA representation of sequence.
        """
        if overlap != 0:
            raise ValueError("HyenaDNA does not chunk sequence.")

        with torch.inference_mode():
            inputs = self.tokenizer(sequence, return_tensors="pt")["input_ids"]
            inputs = inputs.to(self.device)
            hidden_states = self.model(inputs)[0]

        embedding_mean = agg_fn(hidden_states, dim=1)
        return embedding_mean

    def embed_sequence_sixtrack(self, sequence, cds, splice, overlap, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for HyenaDNA.")
