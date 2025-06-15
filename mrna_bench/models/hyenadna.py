from collections.abc import Callable

import torch

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

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use HyenaDNA."
            )

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
        self.max_length = self._get_max_length()

    def _get_max_length(self) -> int:
        """Get maximum sequence length for model."""
        context = self.model_version.split("-")[2]
        if context[-1] == "k":
            return int(context[:-1]) * 1000
        elif context[-1] == "m":
            return int(context[:-1]) * 1000000
        else:
            raise ValueError(
                "Invalid context length in model version. "
                "Expected 'k' or 'm' suffix."
            )

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using HyenaDNA.

        Args:
            sequence: Sequence to embed.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            HyenaDNA representation of sequence.
        """
        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        with torch.inference_mode():
            for c in chunks:
                inputs = self.tokenizer(c, return_tensors="pt")["input_ids"]
                inputs = inputs.to(self.device)

                hidden_states = self.model(inputs)[0]
                embedding_chunks.append(hidden_states)

            hidden_states = torch.cat(embedding_chunks, dim=1)

        embedding_mean = agg_fn(hidden_states, dim=1)
        return embedding_mean

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for HyenaDNA.")
