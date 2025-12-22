from collections.abc import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class OmniGenome(EmbeddingModel):
    """Inference wrapper for OmniGenome.

    OmniGenome is a transformer-based RNA foundation model pretrained
    on plant RNA sequences from the OneKP initiative, using single-nucleotide
    tokenization and ViennaRNA-predicted secondary structures. It is trained
    with three cross-entropy objectives: structure-contextualized masked token
    reconstruction (Str2Seq), sequence-to-structure prediction (Seq2Str), and
    MLM on the sequence.


    Link: https://github.com/yangheng95/OmniGenBench
    """

    max_length = 1024

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize OmniGenome inference wrapper.

        Args:
            model_version: Version of model used. Valid versions: {
                "omnigenome-52m",
                "omnigenome-186m",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use OmniGenome."
            )

        path = ""

        if model_version == "omnigenome-52m":
            path = "yangheng/OmniGenome-52M"
        elif model_version == "omnigenome-186m":
            path = "yangheng/OmniGenome-186M"
        else:
            raise ValueError("Unknown model version.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(device, dtype=torch.float16).eval()

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequence using OmniGenome.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            OmniGenome embedding of sequence with shape (1 x (480 or 720)).
        """
        sequence = sequence.replace("T", "U")
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer(chunk, return_tensors="pt").to(self.device)

            cls_output = self.model(**toks).last_hidden_state
            embedding_chunks.append(cls_output)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for OmniGenome.")
