from collections.abc import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.datasets.dataset_utils import str_to_ohe


class Enformer(EmbeddingModel):
    """Inference wrapper for Enformer.

    Enformer is a deep learning model for predicting functional genomic
    activity from DNA sequence. It uses a transformer-based architecture
    built on convolutional layers and self-attention mechanisms to model
    long-range interactions in the genome. Enformer is trained on 200 kb
    genomic windows and predicts a range of functional readouts across
    multiple human and mouse datasets. Here we use the PyTorch
    implementation of Enformer from EleutherAI, which is based on the
    original Enformer model from DeepMind.

    Link: https://github.com/google-deepmind/deepmind-research
        (Under enformer directory)
    Link: https://github.com/lucidrains/enformer-pytorch
    """

    prediction_window = 114_688  # embedding is of the center 114688 bases
    max_length = 196_608  # can take sequences up to 196608 bases
    bin_size = 128  # embedding is in 128 base bins

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Enformer.

        Args:
            model_version: Version of model used. Valid versions: {
                "enformer-official-rough"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from enformer_pytorch import from_pretrained
        except ImportError:
            raise ImportError("Enformer missing required dependencies.")

        self.model = from_pretrained(
            f'EleutherAI/{model_version}',
            cache_dir=get_model_weights_path()
        ).to(device).eval()

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequence using Enformer, excluding padded regions."""

        def center_padding(seq, length):
            """Center pad sequence to a given length."""
            padding_left = (length - len(seq)) // 2
            padding_right = length - len(seq) - padding_left

            return 'N' * padding_left + seq + 'N' * padding_right, padding_left

        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        with torch.inference_mode():
            for i, chunk in enumerate(chunks):

                padded_chunk, padding_left = center_padding(
                    chunk, self.max_length
                )

                # first OHE sequence chunk
                batch = torch.tensor(
                    str_to_ohe(padded_chunk),
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                _, embedded_chunk = self.model(
                    batch, return_embeddings=True, target_length=-1
                )  # B, L, H

                # extract the portion of the embedding
                # corresponding to original unpadded seq
                start_bin = padding_left // self.bin_size
                end_bin = (
                    padding_left + len(chunk) + (self.bin_size - 1)
                ) // self.bin_size

                embedding = embedded_chunk.permute(0, 2, 1)
                embedding = embedding[:, :, start_bin:end_bin]

                embedding_chunks.append(embedding)

        embedding = torch.cat(embedding_chunks, dim=2)

        aggregate_embedding = agg_fn(embedding, dim=2)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice):
        """Not supported."""
        raise NotImplementedError("Six track not possible with Enformer.")
