from collections.abc import Callable
from functools import partial

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

    max_length = 1026

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize PlantRNAFM inference wrapper.

        Args:
            model_version: Version of model to load.
                    Only "plant_rnafm" is supported.
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use PlantRNAFM."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "yangheng/PlantRNA-FM",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "yangheng/PlantRNA-FM",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(device).eval()

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using PlantRNAFM.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            PlantRNAFM embedding of sequence with shape (1 x 480).
        """
        sequence = sequence.replace("T", "U")
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer(chunk, return_tensors="pt").to(self.device)

            cls_output = self.model(**toks).last_hidden_state
            embedding_chunks.append(cls_output)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for PlantRNAFM.")
