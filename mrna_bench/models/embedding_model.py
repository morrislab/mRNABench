from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any, ClassVar, TypeVar
import re
import warnings

import numpy as np
import torch

from mrna_bench.models.pooling import MeanPool

T = TypeVar("T")


class ModelBehavior(StrEnum):
    """Concrete model behaviors exposed through the benchmark API."""

    EMBEDDING = "embedding"
    CAUSAL_LIKELIHOOD = "causal_likelihood"
    PSEUDO_LIKELIHOOD = "pseudo_likelihood"
    TRACKS = "tracks"


LIKELIHOODS = frozenset({
    ModelBehavior.CAUSAL_LIKELIHOOD,
    ModelBehavior.PSEUDO_LIKELIHOOD,
})


@dataclass(frozen=True)
class TrackOutput:
    """Model-native tracks aligned to an input sequence."""

    values: dict[str, torch.Tensor]
    start: int
    bin_size: int


def discover_layers(
    model: torch.nn.Module,
    patterns: list[str],
) -> list[str]:
    """Find the modules whose names fullmatch any pattern.

    Modules are returned in named_modules() order, which for most
    architectures is also the forward order.

    Args:
        model: Module to search.
        patterns: Fullmatch regexes.

    Returns:
        Matching module paths.
    """
    compiled = [re.compile(pattern) for pattern in patterns]
    return [
        name
        for name, _ in model.named_modules()
        if any(regex.fullmatch(name) for regex in compiled)
    ]


class EmbeddingModel(ABC):
    """Wrapper class for embedding models used to represent sequences."""

    default_version: ClassVar[str]
    valid_versions: ClassVar[list[str]]
    default_attn_implementation: ClassVar[str | None]
    valid_attn_implementations: ClassVar[list[str] | None]
    hookable_layer_patterns: ClassVar[list[str]]
    # Module whose output is the model's final representation (e.g. a final
    # norm), selected with the layer name "final". Defaults to the last
    # hookable layer.
    final_layer_pattern: ClassVar[str | None] = None
    # Linear layers LoRA adapts, matched against the end of each module's
    # path. None means the model does not support LoRA.
    lora_target_modules: ClassVar[list[str] | None] = None
    # Constructor options that change the embeddings, only where they differ
    # from their defaults. Stored in the embedding config.
    model_options: dict = {}
    supported_behaviors: ClassVar[frozenset[ModelBehavior]]
    flash_attn_dtype: ClassVar[torch.dtype] = torch.float16
    sequence_score_scope: ClassVar[str] = "full"
    sequence_score_batch_size: ClassVar[int] = 16
    uses_rna_alphabet: ClassVar[bool] = False
    # Whether <unk> counts as a special token, and so is left out of pooling.
    unk_is_special: ClassVar[bool] = False
    device: torch.device
    model: torch.nn.Module
    tokenizer: Any

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Retrieve shortened name for model version.

        Args:
            model_version: Version of model to fetch short name for.

        Returns:
            Model version with underscores replaced by hyphens.
        """
        return model_version.replace("_", "-")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None = None,
    ):
        """Initialize EmbeddingModel.

        Args:
            model_version: Version of embedding model to use.
            device: PyTorch device to send embedding model.
            attn_implementation: Attention implementation, one of
                valid_attn_implementations. Must be None for models without
                transformer layers.

        Raises:
            ValueError: If model_version is not in valid_versions.
            ValueError: If attn_implementation is not valid for this model.
        """
        if model_version not in self.valid_versions:
            raise ValueError(
                "Invalid model version: {}. Valid versions: {}".format(
                    model_version, self.valid_versions
                )
            )

        valid_impls = self.valid_attn_implementations
        attn_is_set = attn_implementation is not None
        if valid_impls is None:
            if attn_is_set:
                raise ValueError(
                    "{} has no transformer layers; "
                    "attn_implementation must be None, got {!r}.".format(
                        self.__class__.__name__, attn_implementation
                    )
                )
        elif attn_is_set and attn_implementation not in valid_impls:
            raise ValueError(
                "Invalid attn_implementation: {!r}. "
                "Valid implementations for {}: {}".format(
                    attn_implementation, self.__class__.__name__, valid_impls
                )
            )

        self.model_version = model_version
        self.short_name = self.get_model_short_name(model_version)
        self.device = device
        self.attn_implementation = attn_implementation
        self.behaviors = self.behaviors_for_version(model_version)

    @classmethod
    def behaviors_for_version(
        cls,
        model_version: str,
    ) -> frozenset[ModelBehavior]:
        """Return behaviors exposed by a model version.

        Args:
            model_version: Version whose supported behaviors to retrieve.

        Returns:
            Behaviors supported by the requested model version.
        """
        if model_version not in cls.valid_versions:
            raise ValueError(f"Invalid model version: {model_version}")
        return cls.supported_behaviors

    def supports(self, behavior: ModelBehavior | str) -> bool:
        """Return whether this model instance exposes a behavior.

        Args:
            behavior: Model behavior to check.

        Returns:
            True if the loaded model version supports the behavior.
        """
        return ModelBehavior(behavior) in self.behaviors

    def _get_inference_dtype(self) -> torch.dtype:
        """Return the model dtype for the selected attention backend."""
        if self.attn_implementation == "flash_attention_2":
            return self.flash_attn_dtype
        else:
            return torch.float32

    def _set_logits_model(self, model: Any) -> None:
        """Keep an LM head while exposing its backbone for embedding.

        Args:
            model: Language model containing both a backbone and LM head.
        """
        self._logits_model = model
        self.model = model.base_model
        if hasattr(self.model, "pooler"):
            setattr(self.model, "pooler", None)

    @property
    def logits_model(self) -> Any:
        """Model that produces token logits, by default the model itself."""
        return getattr(self, "_logits_model", self.model)

    def tokenize(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize sequences for a forward pass.

        Embedding, extraction and likelihood scoring all tokenize here, so
        they see the same tokens. Sequences are padded to the longest one.

        Args:
            sequences: Nucleotide sequences, or chunks of them.
            cds: Optional CDS tracks aligned to sequences.
            splice: Optional splice tracks aligned to sequences.
            add_special_tokens: Whether to add the model's special tokens,
                e.g. CLS and EOS.

        Returns:
            Tokenizer inputs on the model's device.
        """
        _ = cds, splice
        return self.tokenizer(  # type: ignore[no-any-return]
            [self._tokenizer_text(s) for s in sequences],
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        ).to(self.device)

    def _tokenizer_text(self, sequence: str) -> str:
        """Write a sequence the way the tokenizer reads it.

        Args:
            sequence: Nucleotide sequence in DNA bases.

        Returns:
            Tokenizer input text.
        """
        if self.uses_rna_alphabet:
            return sequence.replace("T", "U")
        else:
            return sequence

    def _tokenize_one(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a single sequence with its optional tracks."""
        return self.tokenize(
            [sequence],
            None if cds is None else [cds],
            None if splice is None else [splice],
            add_special_tokens=add_special_tokens,
        )

    def special_tokens_mask(
        self,
        tokens: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Mark the special tokens in a batch from tokenize().

        Special tokens are those with a special id, such as CLS, EOS and
        padding. <unk> is one only if unk_is_special is set, as an unknown
        base is part of the sequence. Models without a tokenizer have none.

        Args:
            tokens: Output of tokenize().

        Returns:
            Boolean tensor shaped like the input ids, True at special tokens.
        """
        if getattr(self, "tokenizer", None) is None:
            return torch.zeros_like(tokens["attention_mask"], dtype=torch.bool)
        input_ids = tokens["input_ids"]
        special_ids = set(self.tokenizer.all_special_ids)
        if not self.unk_is_special:
            special_ids.discard(self.tokenizer.unk_token_id)
        return torch.isin(
            input_ids,
            torch.tensor(
                sorted(special_ids),
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
        )

    def pooling_mask(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mark the tokens that encode the sequence.

        Args:
            tokens: Output of tokenize().

        Returns:
            Boolean tensor shaped like the input ids, False at padding and
            special tokens.
        """
        mask = ~self.special_tokens_mask(tokens)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            mask &= attention_mask.bool()
        return mask

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask).
        """
        tokens = self.tokenize(chunks)
        hidden_states = self.model(**tokens).last_hidden_state
        return hidden_states, self.pooling_mask(tokens)

    @abstractmethod
    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool(),
    ) -> list[torch.Tensor]:
        """Embed sequences, optionally using cds/splice tracks.

        Args:
            sequences: List of nucleotide sequences to embed (uses DNA bases).
            cds: List of binary encodings of first nucleotide of each codon.
            splice: List of binary encodings of splice site locations.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            One embedding per sequence, shaped by agg_fn.
        """
        pass

    def chunk_sequence(
        self,
        sequence: str,
        chunk_length: int | None,
    ) -> list[str]:
        """Split a sequence into consecutive chunks.

        Args:
            sequence: Sequence to chunk.
            chunk_length: Length of each chunk; the last may be shorter.
                None keeps the sequence whole.

        Returns:
            Sequence chunks.
        """
        if chunk_length is None:
            return [sequence]
        return [
            sequence[i:i + chunk_length]
            for i in range(0, len(sequence), chunk_length)
        ]

    def chunk_tokens(
        self,
        sequence_tokens: list[T],
        chunk_length: int,
    ) -> list[list[T]]:
        """Split a list of tokens into consecutive chunks.

        Args:
            sequence_tokens: Tokens to chunk.
            chunk_length: Length of each chunk; the last may be shorter.

        Returns:
            Token chunks.
        """
        return [
            sequence_tokens[i:i + chunk_length]
            for i in range(0, len(sequence_tokens), chunk_length)
        ]

    def _embed_with_chunking(
        self,
        sequences: list,
        max_chunk_length: int | None,
        embed_fn: Callable[[list], tuple[torch.Tensor, torch.Tensor]],
        agg_fn: Callable = MeanPool(),
        chunk_fn: Callable[[Any], list] | None = None,
    ) -> list[torch.Tensor]:
        """Embed chunks of all sequences in one batch and pool per sequence.

        Args:
            sequences: Sequences to embed, or any items chunk_fn accepts.
            max_chunk_length: Maximum chunk length in nucleotides when
                chunk_fn is not given; None keeps sequences whole.
            embed_fn: Maps a list of chunks to (hidden_states, pooling_mask),
                as _forward_chunks does.
            agg_fn: Function to aggregate token embeddings.
            chunk_fn: Splits one item of sequences into chunks. Defaults to
                chunk_sequence() with max_chunk_length.

        Returns:
            One embedding per sequence.
        """
        chunks = [
            chunk_fn(s) if chunk_fn is not None
            else self.chunk_sequence(s, max_chunk_length)
            for s in sequences
        ]
        flat_chunks = [chunk for seq_chunks in chunks for chunk in seq_chunks]
        hidden_states, pooling_mask = embed_fn(flat_chunks)

        embeddings = []
        start = 0
        for seq_chunks in chunks:
            end = start + len(seq_chunks)
            hidden = hidden_states[start:end].reshape(
                -1, hidden_states.shape[-1]
            )
            mask = pooling_mask[start:end].reshape(-1).bool()
            embeddings.append(agg_fn(hidden[mask]))
            start = end
        return embeddings

    def logits(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
    ) -> list[torch.Tensor]:
        """Return raw per-token logits for likelihood-capable models.

        Args:
            sequences: Nucleotide sequences to score.
            cds: Optional CDS tracks aligned to sequences.
            splice: Optional splice tracks aligned to sequences.

        Returns:
            Per-sequence tensors of raw token logits.
        """
        if not self.behaviors & LIKELIHOODS:
            raise ValueError(
                f"{self.__class__.__name__} does not expose token logits."
            )

        self._validate_score_tracks(sequences, cds, splice)
        logits = []
        for idx, sequence in enumerate(sequences):
            chunk_logits = []
            for chunk, chunk_cds, chunk_splice in self._score_chunks(
                sequence,
                None if cds is None else cds[idx],
                None if splice is None else splice[idx],
            ):
                toks = self._tokenize_one(chunk, chunk_cds, chunk_splice)
                with torch.inference_mode():
                    output = self.logits_model(**toks).logits[0]
                attention_mask = toks.get("attention_mask")
                if attention_mask is not None:
                    output = output[attention_mask[0].bool()]
                chunk_logits.append(output.reshape(-1, output.shape[-1]))
            logits.append(torch.cat(chunk_logits))
        return logits

    def sequence_score(
        self,
        sequences: list[str],
        method: ModelBehavior | str | None = None,
        normalization: str = "mean",
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
    ) -> list[float]:
        """Score complete sequences with a model-native likelihood method.

        Args:
            sequences: Nucleotide sequences to score.
            method: Causal or pseudo-likelihood behavior to use. May be
                omitted when the model supports exactly one likelihood.
            normalization: Return the mean or sum of token log-probabilities.
            cds: Optional CDS tracks aligned to sequences.
            splice: Optional splice tracks aligned to sequences.

        Returns:
            One log-likelihood score per sequence.
        """
        self._validate_score_tracks(sequences, cds, splice)
        supported = self.behaviors & LIKELIHOODS
        if method is None:
            if len(supported) != 1:
                raise ValueError(
                    "method is required when a model supports zero or "
                    "multiple likelihood methods."
                )
            method = next(iter(supported))
        else:
            method = ModelBehavior(method)
        if method not in supported:
            raise ValueError(
                "{} does not support {}. Supported methods: {}".format(
                    self.__class__.__name__,
                    method.value,
                    sorted(item.value for item in supported),
                )
            )
        self._validate_normalization(normalization)

        scorer = (
            self._causal_log_likelihood
            if method == ModelBehavior.CAUSAL_LIKELIHOOD
            else self._pseudo_log_likelihood
        )
        return [
            float(self._reduce(scorer(
                sequence,
                None if cds is None else cds[idx],
                None if splice is None else splice[idx],
            ), normalization).item())
            for idx, sequence in enumerate(sequences)
        ]

    def masked_marginal_llr(
        self,
        reference_sequences: list[str],
        alternate_sequences: list[str],
        normalization: str = "mean",
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
    ) -> list[float]:
        """Compare alleles by masking only tokenizer positions they change.

        Args:
            reference_sequences: Reference allele sequences.
            alternate_sequences: Alternate allele sequences aligned to the
                reference sequences.
            normalization: Return the mean or sum over changed token scores.
            cds: Optional CDS tracks aligned to reference sequences.
            splice: Optional splice tracks aligned to reference sequences.

        Returns:
            Reference-minus-alternate log-likelihood ratios.
        """
        if not self.supports(ModelBehavior.PSEUDO_LIKELIHOOD):
            raise ValueError(
                "{} does not support masked marginal LLR.".format(
                    self.__class__.__name__
                )
            )
        if len(reference_sequences) != len(alternate_sequences):
            raise ValueError(
                "reference_sequences and alternate_sequences must align."
            )
        if any(
            len(reference) != len(alternate)
            for reference, alternate in zip(
                reference_sequences, alternate_sequences
            )
        ):
            raise ValueError(
                "Masked marginal LLR currently supports substitutions only."
            )
        self._validate_normalization(normalization)
        self._validate_score_tracks(reference_sequences, cds, splice)

        llrs = []
        for idx, (reference, alternate) in enumerate(zip(
            reference_sequences, alternate_sequences
        )):
            ref_scores, alt_scores = self._masked_marginal_scores(
                reference,
                alternate,
                None if cds is None else cds[idx],
                None if splice is None else splice[idx],
            )
            ref_total = self._reduce(ref_scores, normalization)
            alt_total = self._reduce(alt_scores, normalization)
            llrs.append(float((ref_total - alt_total).item()))
        return llrs

    @staticmethod
    def _validate_normalization(normalization: str) -> None:
        """Check that normalization is 'mean' or 'sum'."""
        if normalization not in {"mean", "sum"}:
            raise ValueError("normalization must be 'mean' or 'sum'.")

    @staticmethod
    def _reduce(scores: torch.Tensor, normalization: str) -> torch.Tensor:
        """Take the mean or sum of token scores."""
        if normalization == "mean":
            return scores.mean()
        else:
            return scores.sum()

    @staticmethod
    def _validate_score_tracks(
        sequences: list[str],
        cds: list[np.ndarray] | None,
        splice: list[np.ndarray] | None,
    ) -> None:
        """Validate optional per-nucleotide tracks used during scoring.

        Args:
            sequences: Nucleotide sequences being scored.
            cds: Optional CDS tracks aligned to sequences.
            splice: Optional splice tracks aligned to sequences.
        """
        for name, tracks in (("cds", cds), ("splice", splice)):
            if tracks is None:
                continue
            if len(tracks) != len(sequences):
                raise ValueError(
                    "{} must contain one track per sequence.".format(name)
                )
            if any(
                len(track) != len(sequence)
                for sequence, track in zip(sequences, tracks)
            ):
                raise ValueError(
                    "{} track lengths must match sequence lengths.".format(
                        name
                    )
                )

    @staticmethod
    def _slice_tracks(
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
        start: int,
        end: int,
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        """Slice a sequence and its optional tracks."""
        return (
            sequence[start:end],
            None if cds is None else cds[start:end],
            None if splice is None else splice[start:end],
        )

    def _prepare_sequence_for_scoring(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        """Prepare a sequence and tracks before likelihood chunking.

        Args:
            sequence: Nucleotide sequence to prepare.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.

        Returns:
            Prepared sequence and aligned tracks.
        """
        return sequence, cds, splice

    def _score_chunks(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray | None, np.ndarray | None]]:
        """Chunk a sequence and aligned tracks for likelihood scoring.

        Args:
            sequence: Nucleotide sequence to chunk.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.

        Returns:
            Sequence chunks with their aligned CDS and splice tracks.
        """
        sequence, cds, splice = self._prepare_sequence_for_scoring(
            sequence, cds, splice
        )
        chunk_length = getattr(self, "sequence_score_chunk_length", 0)
        if not chunk_length:
            return [(sequence, cds, splice)]
        else:
            return [
                self._slice_tracks(
                    sequence, cds, splice, start, start + chunk_length
                )
                for start in range(0, len(sequence), chunk_length)
            ]

    def _causal_log_likelihood(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> torch.Tensor:
        """Compute next-token log-probabilities for one sequence.

        Chunks after the first start with causal_score_context_length
        nucleotides of the previous chunk, so the boundary tokens are scored
        with context.

        Args:
            sequence: Nucleotide sequence to score.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.

        Returns:
            Log-probability of each token after the first.
        """
        sequence, cds, splice = self._prepare_sequence_for_scoring(
            sequence, cds, splice
        )
        chunk_length = getattr(self, "sequence_score_chunk_length", 0)
        context_length = getattr(self, "causal_score_context_length", 1)
        if chunk_length and context_length >= chunk_length:
            raise ValueError(
                "causal_score_context_length must be smaller than "
                "sequence_score_chunk_length."
            )
        if chunk_length:
            bounds = [(0, chunk_length)] + [
                (start - context_length, start + chunk_length - context_length)
                for start in range(
                    chunk_length,
                    len(sequence),
                    chunk_length - context_length,
                )
            ]
        else:
            bounds = [(0, len(sequence))]
        if len(bounds) > 1:
            warnings.warn(
                "Causal likelihood is computed independently per chunk with "
                "only enough cross-chunk context to score boundary tokens.",
                RuntimeWarning,
            )

        scores: list[torch.Tensor] = []
        for chunk_idx, (start, end) in enumerate(bounds):
            chunk = self._slice_tracks(sequence, cds, splice, start, end)
            toks = self._tokenize_one(*chunk, add_special_tokens=False)
            input_ids = toks["input_ids"]
            if input_ids.shape[1] < 2:
                raise ValueError(
                    "Causal scoring requires at least two tokens per chunk."
                )

            with torch.inference_mode():
                logits = self.logits_model(**toks).logits[:, :-1].float()
                scores.append(torch.log_softmax(logits, dim=-1).gather(
                    -1, input_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1).flatten())
            if chunk_idx:
                # Drop the scores of tokens that only encode the context.
                prefix_ids = self._tokenize_one(
                    *self._slice_tracks(*chunk, 0, context_length),
                    add_special_tokens=False,
                )["input_ids"].reshape(-1)
                common_prefix = 0
                for prefix_id, chunk_id in zip(
                    prefix_ids, input_ids.reshape(-1)
                ):
                    if prefix_id != chunk_id:
                        break
                    common_prefix += 1
                scores[-1] = scores[-1][max(common_prefix - 1, 0):]
        return torch.cat(scores)

    def _mask_token_id(self) -> int:
        """Return the tokenizer's mask token id."""
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Tokenizer does not define a mask token.")
        return mask_token_id

    def _content_positions(self, input_ids: list[int]) -> list[int]:
        """Return the indices of the tokens that are not special tokens."""
        special = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )
        return [
            idx for idx, is_special in enumerate(special) if not is_special
        ]

    def _pseudo_log_likelihood(
        self,
        sequence: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> torch.Tensor:
        """Compute masked-token log-probabilities for one sequence.

        Args:
            sequence: Nucleotide sequence to score.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.

        Returns:
            Log-probability of each token when it alone is masked.
        """
        mask_token_id = self._mask_token_id()
        chunks = self._score_chunks(sequence, cds, splice)
        if len(chunks) > 1:
            warnings.warn(
                "Pseudo-likelihood is computed independently per chunk; "
                "cross-chunk context is not included.",
                RuntimeWarning,
            )

        scores = []
        for chunk in chunks:
            tokenized = self._tokenize_one(*chunk)
            positions = self._content_positions(
                tokenized["input_ids"].reshape(-1).tolist()
            )
            if not positions:
                raise ValueError(
                    "Pseudo-likelihood requires at least one token per chunk."
                )
            scores.extend(self._masked_token_log_probs(
                tokenized, positions, mask_token_id
            ))
        return torch.stack(scores)

    def _masked_marginal_scores(
        self,
        reference: str,
        alternate: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one substitution at every tokenizer position it changes.

        Args:
            reference: Reference allele sequence.
            alternate: Alternate allele sequence.
            cds: Optional CDS track aligned to both sequences.
            splice: Optional splice track aligned to both sequences.

        Returns:
            Masked log-probabilities of the changed reference and alternate
            tokens.
        """
        mask_token_id = self._mask_token_id()
        reference_chunks = self._score_chunks(reference, cds, splice)
        alternate_chunks = self._score_chunks(alternate, cds, splice)
        if len(reference_chunks) != len(alternate_chunks):
            raise ValueError(
                "Reference and alternate scoring chunks do not align."
            )

        reference_scores = []
        alternate_scores = []
        for ref_chunk, alt_chunk in zip(reference_chunks, alternate_chunks):
            ref_tokens = self._tokenize_one(*ref_chunk)
            alt_tokens = self._tokenize_one(*alt_chunk)
            ref_ids = ref_tokens["input_ids"].reshape(-1).tolist()
            alt_ids = alt_tokens["input_ids"].reshape(-1).tolist()
            if ref_ids == alt_ids:
                continue
            ref_changed: set[int] = set()
            alt_changed: set[int] = set()
            for tag, i1, i2, j1, j2 in SequenceMatcher(
                None, ref_ids, alt_ids, autojunk=False
            ).get_opcodes():
                if tag != "equal":
                    ref_changed.update(range(i1, i2))
                    alt_changed.update(range(j1, j2))
            ref_positions = [
                idx for idx in self._content_positions(ref_ids)
                if idx in ref_changed
            ]
            alt_positions = [
                idx for idx in self._content_positions(alt_ids)
                if idx in alt_changed
            ]
            if not ref_positions or not alt_positions:
                raise ValueError(
                    "Tokenizer does not distinguish the reference and "
                    "alternate alleles."
                )
            if len(ref_positions) > 1 or len(alt_positions) > 1:
                warnings.warn(
                    "The variant changes multiple tokenizer tokens; all "
                    "changed tokens will be scored.",
                    RuntimeWarning,
                )
            reference_scores.extend(self._masked_token_log_probs(
                ref_tokens, ref_positions, mask_token_id
            ))
            alternate_scores.extend(self._masked_token_log_probs(
                alt_tokens, alt_positions, mask_token_id
            ))

        if not reference_scores:
            raise ValueError(
                "Tokenizer does not distinguish the reference and "
                "alternate alleles."
            )
        return torch.stack(reference_scores), torch.stack(alternate_scores)

    def _masked_token_log_probs(
        self,
        tokenized: dict[str, torch.Tensor],
        positions: list[int],
        mask_token_id: int,
    ) -> list[torch.Tensor]:
        """Score selected tokens by masking each one independently.

        Args:
            tokenized: Tokenizer inputs for one sequence.
            positions: Token indices to mask and score.
            mask_token_id: Tokenizer ID used for masked positions.

        Returns:
            Log-probability for the original token at each position.
        """
        scores: list[torch.Tensor] = []
        for batch_positions in self.chunk_tokens(
            positions, self.sequence_score_batch_size
        ):
            batch_size = len(batch_positions)
            masked = {
                key: value.repeat(batch_size, *([1] * (value.dim() - 1)))
                for key, value in tokenized.items()
            }
            input_ids = masked["input_ids"].reshape(batch_size, -1)
            rows = torch.arange(batch_size, device=self.device)
            columns = torch.tensor(batch_positions, device=self.device)
            labels = input_ids[rows, columns].clone()
            input_ids[rows, columns] = mask_token_id
            with torch.inference_mode():
                logits = self.logits_model(**masked).logits
                logits = logits.reshape(batch_size, -1, logits.shape[-1])
                scores.extend(torch.log_softmax(
                    logits[rows, columns].float(), dim=-1
                )[rows, labels])
        return scores

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered module paths that extract() can capture.

        Integer layers in extract() index into this list. Found from the
        hookable_layer_patterns class attribute; models whose module order
        differs from their forward order override this property.
        """
        return discover_layers(self.model, self.hookable_layer_patterns)

    @property
    def final_layer(self) -> str:
        """Module path of the model's final representation (layer "final")."""
        matches = []
        if self.final_layer_pattern is not None:
            matches = discover_layers(self.model, [self.final_layer_pattern])
        return matches[0] if matches else self.hookable_layers[-1]

    def _resolve_layer_paths(
        self,
        layers: list[int | str] | None,
    ) -> list[str]:
        """Convert layers argument to a list of module path strings.

        Args:
            layers: None (all hookable_layers), integers (index into
                hookable_layers, supports negative), declared path strings,
                or "final" for final_layer.

        Returns:
            Ordered list of module path strings.
        """
        hookable = self.hookable_layers
        if layers is None:
            return hookable
        resolved = []
        for layer in layers:
            if isinstance(layer, int):
                resolved.append(hookable[layer])
            elif layer == "final":
                resolved.append(self.final_layer)
            elif layer in hookable:
                resolved.append(layer)
            else:
                raise ValueError(
                    "Unknown layer {!r}. Valid layers: {}".format(
                        layer, hookable
                    )
                )
        # "final" can name the same module as an index; capture it once.
        return list(dict.fromkeys(resolved))

    def _register_hooks(
        self,
        layer_paths: list[str],
        detach: bool = True,
    ) -> tuple[list[Any], dict[str, list[torch.Tensor]]]:
        """Register forward hooks that capture module outputs.

        Args:
            layer_paths: Module paths relative to self.model.
            detach: Detach captured tensors from autograd.

        Returns:
            (handles, activations): the hook handles, to remove after the
            forward pass, and each path's captured outputs.
        """
        activations: dict[str, list[torch.Tensor]] = {
            p: [] for p in layer_paths
        }
        handles = []
        for path in layer_paths:
            def hook(
                _module: torch.nn.Module,
                _input: Any,
                output: torch.Tensor | tuple[torch.Tensor, ...],
                name: str = path,
            ) -> None:
                out = output[0] if isinstance(output, tuple) else output
                activations[name].append(out.detach() if detach else out)

            handles.append(
                self.model.get_submodule(path).register_forward_hook(hook)
            )
        return handles, activations

    def _remove_hooks(self, handles: list[Any]) -> None:
        """Remove hooks registered with _register_hooks."""
        for h in handles:
            h.remove()

    def _run_with_layer_capture(
        self,
        layer_paths: list[str],
        forward_fn: Callable[[], Any],
        detach: bool = True,
    ) -> tuple[Any, dict[str, list[torch.Tensor]]]:
        """Run a forward callable while capturing selected module outputs.

        Args:
            layer_paths: Module paths relative to self.model.
            forward_fn: Zero-argument callable that runs the model forward.
            detach: Detach captured tensors from autograd.

        Returns:
            The forward result and captured outputs grouped by module path.
        """
        handles, activations = self._register_hooks(layer_paths, detach)
        try:
            output = forward_fn()
        finally:
            self._remove_hooks(handles)
        return output, activations

    @staticmethod
    def _full_precision_cudnn() -> Any:
        """Run cuDNN operations without TF32 while preserving other flags."""
        return torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            benchmark_limit=torch.backends.cudnn.benchmark_limit,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=False,
        )

    def _warn_batch_size_reproducibility(self, batch_size: int) -> None:
        """Warn when batching can introduce shape-dependent numeric drift."""
        if batch_size > 1:
            warnings.warn(
                "{} batched inference can produce small shape-dependent "
                "floating-point differences. Use batch size 1 for maximum "
                "reproducibility.".format(self.__class__.__name__),
                RuntimeWarning,
                stacklevel=2,
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
        """Extract per-layer token representations and attention scores.

        Args:
            sequences: Sequences to process, in DNA bases.
            cds: Optional CDS tracks, for models that use them.
            splice: Optional splice-site tracks, for models that use them.
            layers: Layers to extract: None for all hookable_layers,
                integers indexing hookable_layers (negative supported),
                hookable layer paths, or "final" for final_layer.
            return_attentions: Also extract attention weights.
            offload_to_cpu: Move captured tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores). hidden_states maps each layer to
            per-sequence lists of per-chunk Tensor(T, D). scores maps each
            layer to per-sequence lists of per-chunk Tensor(H, T, T), or to
            None for layers without attention weights or when
            return_attentions is False.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement extract(). "
            "Add a model-specific extract() override."
        )

    def _standard_hf_extract(
        self,
        sequences: list,
        tokenize_fn: Callable[[list], dict[str, torch.Tensor]] | None = None,
        max_chunk_length: int | None = None,
        layers: list[int | str] | None = None,
        return_attentions: bool = False,
        offload_to_cpu: bool = True,
        chunk_fn: Callable[[Any], list] | None = None,
    ) -> tuple[
        dict[str, list[list[torch.Tensor]]],
        dict[str, list[list[torch.Tensor]] | None],
    ]:
        """Run extract() for a HuggingFace model, one chunk at a time.

        Args:
            sequences: Sequences to chunk, or any items chunk_fn accepts
                (e.g. (sequence, cds) tuples).
            tokenize_fn: Tokenizes a one-element list of chunks. Defaults to
                tokenize().
            max_chunk_length: Chunk length in nucleotides when chunk_fn is
                None.
            layers: Layers to extract; see extract().
            return_attentions: Whether to request attention weights.
            offload_to_cpu: Move captured tensors to CPU after each chunk.
            chunk_fn: Splits one item of sequences into chunks. Defaults to
                chunk_sequence() with max_chunk_length.

        Returns:
            (hidden_states, scores) in the extract() format.
        """
        if self.attn_implementation != "eager":
            if return_attentions:
                warnings.warn(
                    f"Return attention not available with attn_implementation"
                    f"='{self.attn_implementation}'. To extract attentions, "
                    "initialize the model with attn_implementation='eager'."
                )
            return_attentions = False

        if tokenize_fn is None:
            tokenize_fn = self.tokenize
        resolved = self._resolve_layer_paths(layers)
        layer_to_idx = {
            path: index for index, path in enumerate(self.hookable_layers)
        }

        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [[] for _ in sequences] for p in resolved
        }
        score_acc: dict[str, list[list[torch.Tensor]]] = {
            p: [[] for _ in sequences] for p in resolved
        }
        for seq_idx, seq in enumerate(sequences):
            chunks = (
                chunk_fn(seq) if chunk_fn is not None
                else self.chunk_sequence(seq, max_chunk_length)
            )
            for chunk in chunks:
                toks = tokenize_fn([chunk])
                with torch.inference_mode():
                    outputs, captured = self._run_with_layer_capture(
                        resolved,
                        lambda: self.model(
                            **toks,
                            output_hidden_states=False,
                            output_attentions=return_attentions,
                        ),
                    )
                attentions = getattr(outputs, "attentions", None)
                for path in resolved:
                    layer_outputs = captured[path]
                    if len(layer_outputs) != 1:
                        raise RuntimeError(
                            "Expected layer {!r} to run once, captured {} "
                            "outputs.".format(path, len(layer_outputs))
                        )
                    h = layer_outputs[0]
                    if h.dim() == 3 and h.shape[0] == 1:
                        h = h[0]
                    elif h.dim() == 3 and h.shape[1] == 1:
                        h = h[:, 0]
                    if h.dim() != 2:
                        raise RuntimeError(
                            "Expected layer {!r} output with shape (T, D) "
                            "plus one singleton batch dimension, got "
                            "{}.".format(path, tuple(h.shape))
                        )
                    hidden_out[path][seq_idx].append(
                        h.cpu() if offload_to_cpu else h
                    )

                    # The final layer may not be a block with attention.
                    index = layer_to_idx.get(path, len(layer_to_idx))
                    a = None
                    if return_attentions and attentions is not None:
                        if index < len(attentions):
                            a = attentions[index]
                    if a is not None:
                        a = a[0]  # (H, T, T)
                        score_acc[path][seq_idx].append(
                            a.cpu() if offload_to_cpu else a
                        )

        # A path is None iff it never produced attention for any chunk.
        score_out = {
            path: (None if all(len(c) == 0 for c in per_seq) else per_seq)
            for path, per_seq in score_acc.items()
        }
        return hidden_out, score_out

    def get_peft_targets(self) -> list["torch.nn.Module"]:
        """Return modules that LoRA adapters should be applied to.

        Override in subclasses that have multiple independent sub-models
        (e.g. Borzoi's replicate ensemble) or whose backbone lives at a
        non-standard attribute.
        """
        return [self.model]

    def set_peft_targets(
        self, peft_models: list["torch.nn.Module"]
    ) -> None:
        """Replace LoRA target modules after adapter injection.

        Args:
            peft_models: PeftModels returned by get_peft_model(), one per
                target from get_peft_targets().
        """
        self.model = peft_models[0]

    def set_inference_mode(self):
        """Set model to inference mode."""
        self.model.eval()
        self.logits_model.eval()

    def set_train_mode(self):
        """Set model to training mode."""
        self.model.train()
        self.logits_model.train()
