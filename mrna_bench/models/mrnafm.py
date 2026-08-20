from collections.abc import Callable
import warnings

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior


class MRNAFM(EmbeddingModel):
    """Inference wrapper for mRNA-FM.

    mRNA-FM is a transformer-based RNA foundation model pre-trained on coding
    sequences with codon tokenization. It can only accept CDS regions whose
    input length is a multiple of 3.

    Link: https://github.com/ml4bio/RNA-FM/
    """

    default_version = "mRNA-FM"
    valid_versions = ["mRNA-FM"]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    uses_rna_alphabet = True
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })
    sequence_score_scope = "cds"

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        if model_version == "mRNA-FM":
            return "mrna-fm"
        return model_version.replace("_", "-")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize mRNA-FM model.

        Args:
            model_version: Version of mRNA-FM to use. Must be "mRNA-FM".
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
                "Install base_models optional dependency to use mRNA-FM."
            )

        hub_id = "Taykhoom/mRNA-FM"
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        dtype = self._get_inference_dtype()

        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = (self.max_length - 2) * 3

    def _prepare_sequence_for_scoring(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> tuple[str, None, None]:
        """Extract the coding sequence before scoring."""
        _ = splice
        if cds is None:
            raise ValueError("mRNA-FM scoring requires cds tracks.")
        coding_sequence = self.get_cds(sequence, cds).replace("T", "U")
        return coding_sequence, None, None

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding and special tokens (CLS/EOS).
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
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def get_cds(self, sequence: str, cds: np.ndarray) -> str:
        """Get CDS region of sequence.

        CDS must be a multiple of three. For anomalous sequences, returns as
        much of the CDS as possible that is still a multiple of three.

        Args:
            sequence: Sequence to extract CDS region from.
            cds: Binary encoding of first nucleotide of each codon in CDS.

        Returns:
            Sequence of CDS. Returns original sequence if no CDS found with
            truncation to multiple of three.
        """
        if sum(cds) == 0:
            warnings.warn("No CDS found. Returning truncated sequence.")
            return sequence[:len(sequence) - (len(sequence) % 3)]

        first_one_index = np.argmax(cds == 1)
        last_one_index = (len(cds) - 1 - np.argmax(np.flip(cds) == 1)) + 2

        proposed_cds = sequence[first_one_index:last_one_index + 1]

        if len(proposed_cds) % 3 != 0:
            warnings.warn("Irregular CDS. Returning truncated sequence.")
            return proposed_cds[:-(len(proposed_cds) % 3)]

        return proposed_cds

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using mRNA-FM.

        Since mRNA-FM only accepts CDS, uses CDS track to extract CDS sequence
        and generate representation from it. CDS sequence must be a multiple
        of three and is tokenized as non-overlapping codons.

        Args:
            sequences: List of sequences to embed.
            cds: List of binary encodings of first nucleotide of each codon.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
                - default (mean): (1280,)
        """
        _ = splice

        if cds is None:
            raise ValueError("mRNA-FM requires cds to extract coding region.")

        processed_sequences = []
        for i, seq in enumerate(sequences):
            seq = seq.replace("T", "U")
            seq = self.get_cds(seq, cds[i])
            processed_sequences.append(seq)

        return self._embed_with_chunking(
            sequences=processed_sequences,
            max_chunk_length=(self.max_length - 2) * 3,
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
        """Extract per-layer representations from mRNA-FM.

        Since mRNA-FM only accepts CDS, uses CDS track to extract CDS sequence.
        Chunks preserve codon boundaries before HuggingFace tokenization.

        Args:
            sequences: RNA sequences (T or U bases; T->U applied internally).
            cds: CDS tracks used to extract coding region (required).
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _ = splice

        if cds is None:
            raise ValueError("mRNA-FM requires cds to extract coding region.")

        processed_sequences = []
        for i, seq in enumerate(sequences):
            seq = seq.replace("T", "U")
            seq = self.get_cds(seq, cds[i])
            processed_sequences.append(seq)

        def tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
            return self.tokenizer(  # type: ignore[return-value]
                seqs,
                return_tensors="pt",
                padding=False,
            ).to(self.device)

        return self._standard_hf_extract(
            sequences=processed_sequences,
            tokenize_fn=tokenize,
            max_chunk_length=(self.max_length - 2) * 3,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
