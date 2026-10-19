from collections.abc import Callable

import numpy as np
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

    default_version = "omnigenome-186m"
    valid_versions = ["omnigenome-52m", "omnigenome-186m"]

    max_length = 1024

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
            cache_dir=get_model_weights_path(),
            token_dropout=False,
        ).to(device)

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask
            excludes padding and special tokens (CLS/SEP).
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state
        pooling_mask = toks["attention_mask"].clone()

        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            last_pos = seq_lengths[idx] - 1
            pooling_mask[idx, last_pos] = 0

        return hidden_states, pooling_mask

    def _embed_single_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed a single sequence with chunking support.

        Args:
            sequence: Sequence to embed (already converted to RNA).
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embedding with shape (1, hidden_dim).
        """
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        all_hidden = []
        for chunk in chunks:
            hidden_states, pooling_mask = self._forward_chunks([chunk])
            mask = pooling_mask.reshape(-1).bool()
            hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            all_hidden.append(hidden[mask])

        combined_hidden = torch.cat(all_hidden, dim=0)
        return agg_fn(combined_hidden, dim=0).unsqueeze(0)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using OmniGenome.

        Note: OmniGenome processes sequences individually due to architectural
        constraints (LayerNorm, attention masking) that cause different
        embeddings when sequences are batched with padding.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with shape (batch_size, 480 or 720).
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        all_embeddings = []
        for sequence in sequences:
            embedding = self._embed_single_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding)

        return torch.cat(all_embeddings, dim=0)
