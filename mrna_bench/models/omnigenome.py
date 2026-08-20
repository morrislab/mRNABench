from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior


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
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]
    uses_rna_alphabet = True
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize OmniGenome inference wrapper.

        Args:
            model_version: Version of model used. Valid versions: {
                "omnigenome-52m",
                "omnigenome-186m",
            }
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )
        self.max_length = 1024

        try:
            from transformers import (
                AutoConfig,
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
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

        config = AutoConfig.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )
        config.use_flash_attention = (
            self.attn_implementation == "flash_attention_2"
        )

        language_model = AutoModelForMaskedLM.from_pretrained(
            path,
            trust_remote_code=True,
            config=config,
            cache_dir=get_model_weights_path(),
        ).to(device)
        self._set_logits_model(language_model)
        self.sequence_score_chunk_length = self.max_length - 2

        # The 52M model's remote modeling code ignores
        # config.use_flash_attention (unlike the 186M model) and uses
        # FlashAttention whenever the flash_attn package is importable.
        # Force the eager path by clearing the per-layer flash_attn_func,
        # which each attention block checks before using FlashAttention.
        if self.attn_implementation != "flash_attention_2":
            for module in self.model.modules():
                if hasattr(module, "flash_attn_func"):
                    module.flash_attn_func = None  # type: ignore[assignment]

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

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
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
            Embeddings with item shape depending on agg_fn.
            - default (mean): (480,) for `omnigenome-52m`
            - default (mean): (768,) for `omnigenome-186m`
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        embeddings = []
        for sequence in sequences:
            all_hidden = []
            for chunk in self.chunk_sequence(
                sequence, self.max_length - 2
            ):
                hidden_states, pooling_mask = self._forward_chunks([chunk])
                mask = pooling_mask.reshape(-1).bool()
                hidden = hidden_states.reshape(
                    -1, hidden_states.shape[-1]
                )
                all_hidden.append(hidden[mask])
            embeddings.append(agg_fn(torch.cat(all_hidden)))
        return embeddings

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
        """Extract per-layer representations from OmniGenome.

        Note: Processes sequences individually to avoid context-dependent
        embeddings from padding interactions.

        Args:
            sequences: RNA sequences (T or U bases; T->U applied internally).
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            return self.tokenizer(  # type: ignore[return-value]
                seqs,
                return_tensors="pt",
                padding=False,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length - 2,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
