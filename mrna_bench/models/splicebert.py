from collections.abc import Callable

import os
import tarfile

import torch

from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.utils import get_model_weights_path, download_file

# TODO: Change to HF
MODEL_WEIGHT_URL = "https://zenodo.org/records/7995778/files/models.tar.gz"


class SpliceBERT(EmbeddingModel):
    """Inference Wrapper for SpliceBERT.

    SpliceBERT is a transformer-based RNA foundation model trained on 2 million
    vertebrate mRNA sequences using a MLM pretraining objective. Alternative
    versions are trained on only human RNA, and using smaller context windows.

    SpliceBERT 510nt versions strictly use 510nt windows, sequence that is not
    divisible by 510 is truncated.

    Link: https://github.com/biomed-AI/SpliceBERT
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "SpliceBERT.1024nt": "splicebert-v-1024nt",
            "SpliceBERT-human.510nt": "splicebert-v-510nt",
            "SpliceBERT.510nt": "splicebert-h-510nt"
        }
        return short_name_map[model_version]

    def __init__(self, model_version: str, device: torch.device):
        """Initialize SpliceBERT Model.

        Args:
            model_version: Model version to use. Valid versions: {
                "SpliceBERT.1024nt",
                "SpliceBERT-human.510nt",
                "SpliceBERT.510nt"
            }
            device: PyTorch device used by model inference.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use SpliceBERT."
            )

        self.is_sixtrack = False

        # Download all model weights
        weight_path = os.path.join(get_model_weights_path(), "splice-bert")
        os.makedirs(weight_path, exist_ok=True)

        models_parent_dir = os.path.join(weight_path, "models")

        if not os.path.exists(models_parent_dir):
            print("Fetching SpliceBERT weights.")
            dl_path = download_file(MODEL_WEIGHT_URL, str(weight_path))

            with tarfile.open(dl_path) as f:
                f.extractall(weight_path)

            os.remove(dl_path)

        model_path = os.path.join(models_parent_dir, model_version)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)

        self.max_length = int(model_version.split(".")[1].replace("nt", ""))

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using SpliceBERT.

        SpliceBERT's 510nt models only work with a 510nt sequence length input.
        Sequences that are not divisible by 510 will have an overlap between
        the last and second last chunk.

        Args:
            sequence: Sequence to embed.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            SpliceBERT representation of sequence with shape (1 x 512).
        """
        chunks = self.chunk_sequence(sequence, self.max_length)

        # Pad last chunk for 510 nt models
        if self.max_length == 510:
            if len(chunks) == 1 and len(chunks[0]) != 510:
                print(
                    "Warning: SpliceBERT-510nt input must be at least 510nts."
                    "Embedding may not work correctly."
                )
            elif len(chunks[-1]) != 510:
                overlap = 510 - len(chunks[-1])
                chunks[-1] = chunks[-2][-overlap:] + chunks[-1]
                assert len(chunks[-1]) == 510

        embedding_chunks = []

        for chunk in chunks:
            chunk_in = " ".join(list(chunk))
            input_ids = self.tokenizer.encode(chunk_in)
            input_ids = torch.as_tensor(input_ids)
            input_ids = input_ids.unsqueeze(0).to(self.device)

            last_hidden_state = self.model(input_ids).last_hidden_state

            embedding_chunks.append(last_hidden_state)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for SpliceBERT.")
