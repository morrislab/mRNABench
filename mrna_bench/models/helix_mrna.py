from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


class HelixmRNA(EmbeddingModel):
    """Inference wrapper for Helix-mRNA.

    Helix-mRNA is a RNA foundation model trained using a Mamba2 and transformer
    hybrid backbone. Helix-mRNA is pre-trained on 26M mRNAs from diverse
    eukaryotic and viral species.

    Link: https://github.com/helicalAI/helical
    """

    default_version = "helix-mrna"
    valid_versions = ["helix-mrna"]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Helix-mRNA model.

        Args:
            model_version: Must be "helix-mrna".
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("Helix-mRNA missing required dependencies.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Taykhoom/Helix-mRNA-Wrapper",
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        dtype = self._get_inference_dtype()

        self.model = AutoModel.from_pretrained(
            "Taykhoom/Helix-mRNA-Wrapper",
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(self.device)

        self.max_length = self.tokenizer.model_max_length

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and special tokens (CLS/SEP).
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_special_tokens_mask=True,
        ).to(self.device)

        special_tokens_mask = toks["special_tokens_mask"]
        attention_mask = 1 - special_tokens_mask

        hidden_states = self.model(
            input_ids=toks["input_ids"],
            attention_mask=attention_mask,
        ).last_hidden_state

        pooling_mask = attention_mask.clone()

        return hidden_states, pooling_mask

    def _tokenize_cds(self, sequence: str, cds: np.ndarray) -> str:
        """Convert sequence to Helix-mRNA vocab by inserting 'E' tokens."""
        modified_sequence = ""
        for i in range(len(sequence)):
            if cds[i] == 1:
                modified_sequence += "E"
            modified_sequence += sequence[i]

        return modified_sequence

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Batch embed sequences using Helix-mRNA.

        If cds is provided, inserts 'E' tokens at the start of each codon
        to use Helix-mRNA's codon-aware vocabulary.

        Args:
            sequences: List of sequences to embed.
            cds: List of binary encodings of first nucleotide of each codon.
            splice: Unused.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embeddings with item shape depending on agg_fn.
             - default (mean): (256,)
        """
        _ = splice  # Unused

        if cds is not None:
            sequences = [
                self._tokenize_cds(seq, c).upper().replace("T", "U")
                for seq, c in zip(sequences, cds)
            ]
        else:
            sequences = [s.upper().replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 1,  # Account for SEP token
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )

    def extract(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        layers: list[int | str] | None = None,
        return_attentions: bool = False,
        offload_to_cpu: bool = True,
    ) -> tuple[
        dict[str, list[list[torch.Tensor]]],
        dict[str, list[list[torch.Tensor]] | None],
    ]:
        """Extract per-layer representations from Helix-mRNA.

        If cds is provided, inserts 'E' tokens at the start of each codon
        to use Helix-mRNA's codon-aware vocabulary.

        Attention extraction requires the model to be loaded with
        attn_implementation="eager". Flash Attention 2 does not materialise
        the (H, T, T) attention matrix; scores will be None for all layers
        when flash_attention_2 is active. Only layers whose block type is
        "attention" (only layer 3 in the default "M+M*M+M+" config) can
        produce attention weights; Mamba and MLP layers yield None.

        Args:
            sequences: RNA sequences (T or U bases; T->U applied internally).
            cds: Optional CDS tracks; if provided, codon 'E' tokens are added.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
            scores[layer] is None for Mamba and MLP layers regardless of
            return_attentions.
        """
        _ = splice

        if cds is not None:
            sequences = [
                self._tokenize_cds(seq, c).upper().replace("T", "U")
                for seq, c in zip(sequences, cds)
            ]
        else:
            sequences = [s.upper().replace("T", "U") for s in sequences]

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            toks = self.tokenizer(  # type: ignore[return-value]
                seqs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            ).to(self.device)
            toks["attention_mask"] = 1 - toks.pop("special_tokens_mask")
            return toks

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length - 1,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
