from collections.abc import Callable
from functools import partial

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


class AIDORNA(EmbeddingModel):
    """Inference wrapper for AIDO.RNA.

    AIDO.RNA is a transformer-based RNA foundation model. It is trained using
    masked language modelling on 42 million non-coding RNA sequences, with
    domain adaptation models available for protein coding sequences.

    Link: https://github.com/genbio-ai/ModelGenerator
    """

    max_length = 1024

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize AIDO.RNA.

        Args:
            model_version: Version of model used. Valid versions: {
                "aido_rna_1b600m",
                "aido_rna_1b600m_cds",
                "aido_rna_650m",
                "aido_rna_650m_cds",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use AIDO.RNA."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Taykhoom/AIDO-RNA-Wrapper",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        self.model = AutoModel.from_pretrained(
            "Taykhoom/AIDO-RNA-Wrapper",
            trust_remote_code=True,
            base_model=model_version,
            cache_dir=get_model_weights_path(),
        ).to(device)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using AIDO.RNA.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            AIDO.RNA embedding of sequence with shape (1 x H).
            H is:
                1280 for 650M model
                2048 for 1.6B model
        """
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        for i, chunk in enumerate(chunks):
            batch = self.tokenizer(
                chunk,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.device)

            t_keys = ["input_ids", "attention_mask"]

            # Strip start and stop tokens from all but first and last chunk
            # if only one chunk, do nothing
            if len(chunks) != 1:
                if i == 0:
                    for k in t_keys:
                        batch[k] = batch[k][:, :-1]
                elif i == len(chunks) - 1:
                    for k in t_keys:
                        batch[k] = batch[k][:, 1:]
                else:
                    for k in t_keys:
                        batch[k] = batch[k][:, 1:-1]

            embedded_chunk = self.model(**batch).last_hidden_state
            embedding_chunks.append(embedded_chunk)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not possible with AIDO.RNA.")
