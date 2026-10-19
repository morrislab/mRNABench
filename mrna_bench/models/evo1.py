from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class Evo1(EmbeddingModel):
    """Inference wrapper for Evo1.

    Evo1 is a StripedHyena-based DNA foundation model trained on the
    OpenGenome dataset using an autoregressive scheme at single nucleotide,
    byte level resolution. Owing to its StripedHyena backbone, it has a near
    linear scaling of compute and memory relative to its context window.

    Checkpoints are loaded from HuggingFace ports of the original releases,
    which bundle their own model code.

    Note: StripedHyena's convolutions don't fully isolate sequences in batched
    mode even with padding mask. This implementation uses single-sequence
    processing for consistency.

    Link: https://github.com/evo-design/evo
    """

    default_version = "evo-1.5-8k-base"
    valid_versions = [
        "evo-1.5-8k-base",
        "evo-1-8k-base",
        "evo-1-131k-base",
    ]

    # Ported repo and pinned revision for each version.
    version_to_repo = {
        "evo-1.5-8k-base": (
            "Taykhoom/Evo1-1.5-7B-8K",
            "5b63c84c2b8473d425379dad5becb1411e19c782",
        ),
        "evo-1-8k-base": (
            "Taykhoom/Evo1-1-7B-8K",
            "ae7474e07803c37f2e5111f0a054669902e7dc35",
        ),
        "evo-1-131k-base": (
            "Taykhoom/Evo1-1-7B-131K",
            "f997f205f1a21fd4580a97562b44760317b55978",
        ),
    }

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Evo1.

        Args:
            model_version: Version of model used. Valid versions: {
                "evo-1.5-8k-base",
                "evo-1-8k-base",
                "evo-1-131k-base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use Evo1."
            )

        repo_id, revision = self.version_to_repo[model_version]
        hub_kwargs = {
            "revision": revision,
            "cache_dir": get_model_weights_path(),
            "trust_remote_code": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **hub_kwargs)
        self.model = AutoModel.from_pretrained(
            repo_id,
            attn_implementation="flash_attention_2",
            **hub_kwargs,
        ).to(device)
        self.max_length = self.tokenizer.model_max_length

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed a single sequence using Evo1.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo1 embedding with shape (1, 4096).
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer([chunk], return_tensors="pt").to(self.device)
            outputs = self.model(**toks)
            embedding_chunks.append(outputs.last_hidden_state[0])

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
        """Embed sequences using Evo1.

        Processes sequences one at a time due to StripedHyena's architectural
        limitation with padding (convolutions don't fully isolate sequences).

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo1 embeddings with shape (batch_size, 4096).
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding)

        return torch.cat(all_embeddings, dim=0)
