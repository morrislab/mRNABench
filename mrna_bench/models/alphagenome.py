from collections.abc import Callable
import math

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.datasets.dataset_utils import str_to_ohe
from mrna_bench.models.embedding_model import EmbeddingModel


class AlphaGenome(EmbeddingModel):
    """Inference wrapper for AlphaGenome.

    AlphaGenome is a deep learning model for predicting functional genomic
    activity from DNA sequence. It combines convolutional layers with
    self-attention to model long-range interactions over up to 1 Mb of
    sequence, and predicts functional readouts across human and mouse
    datasets. This uses the PyTorch port of the DeepMind weights; the weights
    are subject to Google DeepMind's AlphaGenome Model Terms
    (https://deepmind.google.com/science/alphagenome/model-terms).

    Link: https://github.com/google-deepmind/alphagenome_research
    Link: https://github.com/genomicsxai/alphagenome-pytorch
    """

    default_version = "alphagenome"
    valid_versions = ["alphagenome"]

    max_length = 1_048_576

    weights_repo = "gtca/alphagenome_pytorch"
    weights_revision = "b01c0ffa73e07c053491f3b5ea8bcf67d93b9920"

    # Organism index for the human head.
    organism_index = 0

    def __init__(self, model_version: str, device: torch.device):
        """Initialize AlphaGenome.

        Args:
            model_version: Version of model used. Valid versions: {
                "alphagenome"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from alphagenome_pytorch import AlphaGenome as AlphaGenomeModel
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use AlphaGenome."
            )

        weights_file = hf_hub_download(
            self.weights_repo,
            "model_all_folds.safetensors",
            revision=self.weights_revision,
            cache_dir=get_model_weights_path(),
        )
        self.model = AlphaGenomeModel.from_pretrained(
            weights_file, device=device
        )

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed a single sequence using AlphaGenome.

        Each chunk is right-padded with N to a multiple of 2048 bases, the
        model's resolution, and the padding is sliced off its 1 bp
        embeddings before aggregation.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            AlphaGenome embedding with shape (1, 1536).
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        for chunk in chunks:
            padded_length = math.ceil(len(chunk) / 2048) * 2048
            padded_chunk = chunk + "N" * (padded_length - len(chunk))

            ohe = torch.tensor(
                str_to_ohe(padded_chunk), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            organism = torch.tensor(
                [self.organism_index], dtype=torch.long, device=self.device
            )

            result = self.model.encode(ohe, organism, resolutions=(1,))
            embedding_chunks.append(result["embeddings_1bp"][0, :len(chunk)])

        return agg_fn(
            torch.cat(embedding_chunks, dim=0), dim=0
        ).float().unsqueeze(0)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using AlphaGenome.

        Processes sequences one at a time due to memory constraints at long
        sequence lengths.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            AlphaGenome embeddings with shape (batch_size, 1536).
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            all_embeddings.append(self.embed_sequence(sequence, agg_fn=agg_fn))

        return torch.cat(all_embeddings, dim=0)
