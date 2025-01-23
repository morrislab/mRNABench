from collections.abc import Callable

import torch

from mrna_bench.models.embedding_model import EmbeddingModel


class AIDORNA(EmbeddingModel):
    """Inference wrapper for AIDO.RNA.

    AIDO.RNA is a transformer-based RNA foundation model. It is trained using
    masked language modelling on 42 million non-coding RNA sequences, with
    domain adaptation models available for protein coding sequences.

    Link: https://github.com/genbio-ai/ModelGenerator
    """

    MAX_LENGTH = 1024

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("rna_", "").replace("_", "-")

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
        from modelgenerator.tasks import Embed

        model = Embed.from_config({"model.backbone": model_version}).eval()

        self.model = model.to(device)

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using AIDO.RNA.

        Args:
            sequence: Sequence to be embedded.
            overlap: Number of tokens overlapping between chunks.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            AIDO.RNA embedding of sequence with shape (1 x H).
        """
        chunks = self.chunk_sequence(sequence, self.MAX_LENGTH - 2, overlap)

        embedding_chunks = []

        for i, chunk in enumerate(chunks):
            batch = self.model.transform({"sequences": [chunk]})

            t_keys = ["special_tokens_mask", "input_ids", "attention_mask"]

            # Strip start and stop tokens from all but first and last chunk
            if i == 0:
                for k in t_keys:
                    batch[k] = batch[k][:, :-1]
            elif i == len(chunks) - 1:
                for k in t_keys:
                    batch[k] = batch[k][:, 1:]
            else:
                for k in t_keys:
                    batch[k] = batch[k][:, 1:-1]

            embedded_chunk = self.model(batch)
            embedding_chunks.append(embedded_chunk)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice):
        """Not supported."""
        raise NotImplementedError("Six track not possible with AIDO.RNA.")
