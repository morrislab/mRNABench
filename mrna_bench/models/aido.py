from collections.abc import Callable

import torch

from mrna_bench.models.embedding_model import EmbeddingModel


class AIDORNA(EmbeddingModel):
    MAX_LENGTH = 1024
    SHORT_NAME_DICT = {
        "aido_rna_1b600m_cds": "aido-1b600m-cds",
        "aido_rna_1b600m": "aido-1b600m",
        "aido_rna_650m": "aido-650m",
    }

    def __init__(self, model_version, device):
        super().__init__(model_version, device)

        from modelgenerator.tasks import Embed
        model = Embed.from_config({"model.backbone": model_version}).eval()

        self.model = model.to(device)

    def get_model_short_name(self):
        return self.SHORT_NAME_DICT[self.model_version]

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        chunks = self.chunk_sequence(sequence, self.MAX_LENGTH - 2, overlap)

        embedding = []

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
            embedding.append(embedded_chunk)

        embedding = torch.cat(embedding, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice):
        raise NotImplementedError("Six track not possible with AIDO.RNA.")
