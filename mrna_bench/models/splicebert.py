from collections.abc import Callable

import numpy as np
import torch

from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior
from mrna_bench.utils import get_model_weights_path


class SpliceBERT(EmbeddingModel):
    """Inference Wrapper for SpliceBERT.

    SpliceBERT is a transformer-based RNA foundation model trained on 2 million
    vertebrate mRNA sequences using a MLM pretraining objective. Alternative
    versions are trained on only human RNA, and using smaller context windows.

    SpliceBERT-510nt versions strictly use 510nt windows, sequence that is not
    divisible by 510 is truncated.

    Link: https://github.com/biomed-AI/SpliceBERT
    """

    default_version = "SpliceBERT-1024nt"
    valid_versions = [
        "SpliceBERT-1024nt",
        "SpliceBERT-510nt",
        "SpliceBERT-human-510nt",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "SpliceBERT-1024nt": "splicebert-v-1024nt",
            "SpliceBERT-510nt": "splicebert-v-510nt",
            "SpliceBERT-human-510nt": "splicebert-h-510nt",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize SpliceBERT Model.

        Args:
            model_version: Model version to use. Valid versions: {
                "SpliceBERT-1024nt",
                "SpliceBERT-510nt",
                "SpliceBERT-human-510nt"
            }
            device: PyTorch device used by model inference.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use SpliceBERT."
            )

        hub_id = "Taykhoom/{}".format(model_version)
        cache_dir = get_model_weights_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        dtype = self._get_inference_dtype()
        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            attn_implementation=self.attn_implementation,
            cache_dir=cache_dir,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)

        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize spaced nucleotides for masked scoring."""
        _ = cds, splice
        return self.tokenizer(  # type: ignore[no-any-return]
            " ".join(sequence),
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

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
        agg_fn: Callable = EmbeddingModel.mean_pool
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
        agg_fn: Callable = EmbeddingModel.mean_pool
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
        agg_fn: Callable = EmbeddingModel.mean_pool
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
        """Extract per-layer representations from SpliceBERT.

        Uses single-nucleotide spaced tokenization (" ".join(list(seq))).

        Args:
            sequences: RNA/DNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            spaced = [" ".join(list(s)) for s in seqs]
            return self.tokenizer(  # type: ignore[return-value]
                spaced,
                return_tensors="pt",
                padding=False,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=sequences,
            tokenize_fn=tokenize,
            max_chunk_length=self.max_length,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
