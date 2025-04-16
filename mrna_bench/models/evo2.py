from collections.abc import Callable

import torch
import os 

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


class Evo2(EmbeddingModel):
    """Inference wrapper for Evo2.

    Evo2 is a StripedHyena2-based DNA foundation model trained on the OpenGenome2
    dataset using an autoregressive scheme at single nucleotide resolution. Owing
    to its StripedHyena2 backbone, it has an ultra long context window. The `base`
    variants can handle sequences up to 8192 nucleotides in length while the
    larger variants can handle sequences up to 1 million nucleotides in length.

    Link: https://github.com/ArcInstitute/evo2
    """

    max_length = 8_192

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Evo2.

        Args:
            model_version: Version of model used. Valid versions: {
                "evo2_40b",
                "evo2_7b",
                "evo2_40b_base",
                "evo2_7b_base",
                "evo2_1b_base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            os.environ['HF_HUB_CACHE'] = get_model_weights_path()
            from evo2 import Evo2
        except ImportError:
            raise ImportError("Evo2 must be installed to use this model.")

        self.model = Evo2(model_version)
        self.tokenizer = self.model.tokenizer.tokenize
        
        all_prenorms = [name.strip('.scale') for name,_ in self.model.named_parameters() if "pre_norm" in name]

        # we will only take the middle and last layer output for simplicity
        self.embedding_layers = [all_prenorms[len(all_prenorms)//2], 'norm']

        if model_version in ["evo2_40b", "evo2_7b"]:
            max_length = 1_000_000

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using Evo2.

        Args:
            sequence: Sequence to be embedded.
            overlap: Number of tokens overlapping between chunks.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo2 embedding of sequence with shape (1 x H).
        """
        chunks = self.chunk_sequence(sequence, self.max_length - 2, overlap)

        embedding_chunks = []

        with torch.inference_mode():

            for i, chunk in enumerate(chunks):
                
                input_ids = torch.tensor(
                    self.tokenizer(chunk), 
                    dtype=torch.int
                ).unsqueeze(0).to(self.device)

                _, embeddings = self.model(
                    input_ids = input_ids, 
                    return_embeddings=True, 
                    layer_names=self.embedding_layers
                )

                embedding_chunks.append(embeddings)

        aggregate_embeddings = []

        # embedding is of type bfloat16, need to convert to float32
        # since numpy does not support bfloat16
        for layer_name in sorted(self.embedding_layers):
            aggregate_embeddings.append(torch.mean(torch.cat([embedding_chunks[i][layer_name] for i in range(len(embedding_chunks))], dim=1), dim=1).float().cpu())

        aggregate_embedding = torch.vstack(aggregate_embeddings)

        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice):
        """Not supported."""
        raise NotImplementedError("Six track not possible with Evo2.")
