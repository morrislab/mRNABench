from collections.abc import Callable
from functools import partial

import os
import tarfile

import numpy as np
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

    default_version = "SpliceBERT.1024nt"
    valid_versions = [
        "SpliceBERT.1024nt",
        "SpliceBERT-human.510nt",
        "SpliceBERT.510nt",
    ]

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            clean_up_tokenization_spaces=True,
        )
        self.model = AutoModel.from_pretrained(model_path).to(device)

        self.max_length = int(model_version.split(".")[1].replace("nt", ""))

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask) tensors.
        """
        spaced_chunks = [" ".join(list(chunk)) for chunk in chunks]

        toks = self.tokenizer(
            spaced_chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state

        pooling_mask = toks["attention_mask"].clone()
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def _embed_1024(
        self,
        sequences: list[str],
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using 1024nt model.

        Args:
            sequences: List of sequences to embed.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            SpliceBERT embeddings with shape (batch_size, 512).
        """
        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )

    def _embed_510(
        self,
        sequences: list[str],
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using 510nt model with overlap handling.

        Args:
            sequences: List of sequences to embed.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (512,)
        """
        all_chunks = []
        chunk_counts = []

        for seq in sequences:
            chunks = self.chunk_sequence(seq, 510)
            if len(chunks) == 1 and len(chunks[0]) != 510:
                print(
                    "Warning: SpliceBERT-510nt input must be at least 510nts. "
                    "Embedding may not work correctly."
                )
            elif len(chunks) > 1 and len(chunks[-1]) != 510:
                overlap = 510 - len(chunks[-1])
                chunks[-1] = chunks[-2][-overlap:] + chunks[-1]
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        hidden_states, pooling_mask = self._forward_chunks(all_chunks)

        seq_embeddings = []
        chunk_ptr = 0

        for num_chunks in chunk_counts:
            seq_hidden = hidden_states[chunk_ptr:chunk_ptr + num_chunks]
            seq_mask = pooling_mask[chunk_ptr:chunk_ptr + num_chunks]

            hidden = seq_hidden.reshape(-1, seq_hidden.shape[-1])
            mask = seq_mask.reshape(-1).bool()

            masked_hidden = hidden[mask]
            seq_embeddings.append(agg_fn(masked_hidden, dim=0))

            chunk_ptr += num_chunks

        return seq_embeddings

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using SpliceBERT.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
                - default (mean): (512,)
        """
        _, _ = cds, splice

        if self.max_length == 510:
            return self._embed_510(sequences, agg_fn)
        else:
            return self._embed_1024(sequences, agg_fn)
