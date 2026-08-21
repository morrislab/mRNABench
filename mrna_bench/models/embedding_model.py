from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable
import re
import warnings

import numpy as np
import torch

T = TypeVar("T")


def mean_pool(
    hidden_states: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Mean-pool token states in FP32.

    Args:
        hidden_states: Token representations to pool.
        dim: Dimension containing the tokens to average.

    Returns:
        Mean-pooled representations in FP32.
    """
    # there can be perf differences if using a FA2
    # model with FP16 hidden states, so we cast to FP32
    return hidden_states.float().mean(dim=dim)


class ModelBehavior(StrEnum):
    """Concrete model behaviors exposed through the benchmark API."""

    EMBEDDING = "embedding"
    CAUSAL_LIKELIHOOD = "causal_likelihood"
    PSEUDO_LIKELIHOOD = "pseudo_likelihood"
    TRACKS = "tracks"


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
    """Discover hookable module paths by regex, in registration order.

    Returns every module whose name fullmatches *any* of the given
    patterns, preserving the model's ``named_modules()`` traversal order
    (i.e. submodule registration order). For most architectures this equals
    the forward / ``hidden_states`` order, so a model only needs to declare
    its block patterns rather than reimplement this loop. Models whose
    registration order differs from their execution order (e.g. Evo, whose
    final ``norm`` is registered before the blocks but applied after them)
    must override the ``hookable_layers`` property instead.

    Args:
        model: Module to search.
        patterns: List of fullmatch regexes. A module is included if it
            matches any of them. Each model declares its own patterns via
            the ``hookable_layer_patterns`` class attribute; there is no
            shared default.

    Returns:
        List of matching module path strings in ``named_modules()`` order.
    """
    compiled = [re.compile(pattern) for pattern in patterns]
    return [
        name
        for name, _ in model.named_modules()
        if any(regex.fullmatch(name) for regex in compiled)
    ]


@runtime_checkable
class SupportsEmbedding(Protocol):
    """Protocol defining the interface for embedding models."""

    device: torch.device
    model: torch.nn.Module

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None,
        splice: list[np.ndarray] | None,
        agg_fn: Callable,
    ) -> list[torch.Tensor]:
        """Embed sequences, optionally using cds/splice tracks."""
        ...


class EmbeddingModel(SupportsEmbedding, ABC):
    """Wrapper class for embedding models used to represent sequences."""

    mean_pool = staticmethod(mean_pool)
    default_version: ClassVar[str]
    valid_versions: ClassVar[list[str]]
    default_attn_implementation: ClassVar[str | None]
    valid_attn_implementations: ClassVar[list[str] | None]
    hookable_layer_patterns: ClassVar[list[str]]
    supported_behaviors: ClassVar[frozenset[ModelBehavior]]
    flash_attn_dtype: ClassVar[torch.dtype] = torch.float16
    sequence_score_scope: ClassVar[str] = "full"
    sequence_score_batch_size: ClassVar[int] = 16
    uses_rna_alphabet: ClassVar[bool] = False
    model: torch.nn.Module
    tokenizer: Any

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Retrieve shortened name for model version.

        Override in subclass if the version name needs custom transformation.
        By default, replaces underscores with hyphens.

        Args:
            model_version: Version of model to fetch short name for.

        Returns:
            Shortened name of model version.
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
            attn_implementation: Attention implementation to use.  Subject
                to what each model supports (see
                ``valid_attn_implementations``). For models
                without transformer layers ``valid_attn_implementations``
                is ``None`` and this argument must also be ``None``.

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
        return torch.float32

    def _set_logits_model(self, model: Any) -> None:
        """Keep an LM head while exposing its backbone for embedding.

        Args:
            model: Language model containing both a backbone and LM head.
        """
        self.logits_model = model
        self.model = model.base_model
        if hasattr(self.model, "pooler"):
            setattr(self.model, "pooler", None)

    def _get_logits_model(self) -> Any:
        """Return the model that produces token logits."""
        return getattr(self, "logits_model", self.model)

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
        likelihoods = {
            ModelBehavior.CAUSAL_LIKELIHOOD,
            ModelBehavior.PSEUDO_LIKELIHOOD,
        }
        if not self.behaviors.intersection(likelihoods):
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
                tokenized = self._tokenize_for_logits(
                    chunk,
                    chunk_cds,
                    chunk_splice,
                )
                toks = {
                    key: value.to(self.device)
                    for key, value in tokenized.items()
                }
                with torch.inference_mode():
                    output = self._get_logits_model()(**toks).logits[0]
                attention_mask = toks.get("attention_mask")
                if attention_mask is not None:
                    output = output[attention_mask[0].bool()]
                chunk_logits.append(
                    output.reshape(-1, output.shape[-1])
                )
            logits.append(torch.cat(chunk_logits))
        return logits

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenize one sequence for a language-model forward pass.

        Args:
            sequence: Nucleotide sequence to tokenize.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.
            add_special_tokens: Whether to add model-specific special tokens.

        Returns:
            Tokenizer inputs ready for a model forward pass.
        """
        _ = cds, splice
        if self.uses_rna_alphabet:
            sequence = sequence.replace("T", "U")
        return self.tokenizer(  # type: ignore[no-any-return]
            sequence,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

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
        supported = self.behaviors.intersection({
            ModelBehavior.CAUSAL_LIKELIHOOD,
            ModelBehavior.PSEUDO_LIKELIHOOD,
        })
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
        if normalization not in {"mean", "sum"}:
            raise ValueError("normalization must be 'mean' or 'sum'.")

        scorer = (
            self._causal_log_likelihood
            if method == ModelBehavior.CAUSAL_LIKELIHOOD
            else self._pseudo_log_likelihood
        )
        return [
            scorer(
                sequence,
                normalization,
                None if cds is None else cds[idx],
                None if splice is None else splice[idx],
            )
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
        if normalization not in {"mean", "sum"}:
            raise ValueError("normalization must be 'mean' or 'sum'.")
        self._validate_score_tracks(reference_sequences, cds, splice)

        return [
            self._masked_marginal_llr(
                reference,
                alternate,
                normalization,
                None if cds is None else cds[idx],
                None if splice is None else splice[idx],
            )
            for idx, (reference, alternate) in enumerate(zip(
                reference_sequences, alternate_sequences
            ))
        ]

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
        return [
            (
                sequence[start:start + chunk_length],
                None if cds is None else cds[start:start + chunk_length],
                (
                    None if splice is None
                    else splice[start:start + chunk_length]
                ),
            )
            for start in range(0, len(sequence), chunk_length)
        ]

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

    def _causal_log_likelihood(
        self,
        sequence: str,
        normalization: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> float:
        """Compute next-token log-likelihood for one sequence.

        Args:
            sequence: Nucleotide sequence to score.
            normalization: Return the mean or sum over token scores.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.

        Returns:
            Next-token log-likelihood for the sequence.
        """
        scores: list[torch.Tensor] = []
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
        starts = (
            [0]
            if not chunk_length
            else [0] + list(range(
                chunk_length,
                len(sequence),
                chunk_length - context_length,
            ))
        )
        chunks = [
            (
                sequence[
                    max(start - context_length, 0):
                    start + (
                        chunk_length - context_length
                        if start else chunk_length or len(sequence)
                    )
                ],
                None if cds is None else cds[
                    max(start - context_length, 0):
                    start + (
                        chunk_length - context_length
                        if start else chunk_length or len(sequence)
                    )
                ],
                None if splice is None else splice[
                    max(start - context_length, 0):
                    start + (
                        chunk_length - context_length
                        if start else chunk_length or len(sequence)
                    )
                ],
            )
            for start in starts
        ]
        if len(chunks) > 1:
            warnings.warn(
                "Causal likelihood is computed independently per chunk with "
                "only enough cross-chunk context to score boundary tokens.",
                RuntimeWarning,
            )
        for chunk_idx, (chunk, chunk_cds, chunk_splice) in enumerate(chunks):
            prefix = chunk[:context_length] if chunk_idx else ""
            prefix_cds = (
                chunk_cds[:context_length]
                if chunk_idx and chunk_cds is not None
                else None
            )
            prefix_splice = (
                chunk_splice[:context_length]
                if chunk_idx and chunk_splice is not None
                else None
            )
            tokenized = self._tokenize_for_logits(
                chunk,
                chunk_cds,
                chunk_splice,
                add_special_tokens=False,
            )
            toks = {
                key: value.to(self.device)
                for key, value in tokenized.items()
            }
            input_ids = toks["input_ids"]
            if input_ids.shape[1] < 2:
                raise ValueError(
                    "Causal scoring requires at least two tokens per chunk."
                )

            with torch.inference_mode():
                outputs = self._get_logits_model()(**toks)
                logits = outputs.logits[:, :-1].float()
                labels = input_ids[:, 1:]
                scores.append(torch.log_softmax(logits, dim=-1).gather(
                    -1, labels.unsqueeze(-1)
                ).squeeze(-1).flatten())
            if prefix:
                prefix_ids = self._tokenize_for_logits(
                    prefix,
                    prefix_cds,
                    prefix_splice,
                    add_special_tokens=False,
                )["input_ids"].reshape(-1)
                combined_ids = input_ids.reshape(-1)
                common_prefix = 0
                for prefix_id, combined_id in zip(
                    prefix_ids, combined_ids
                ):
                    if prefix_id != combined_id:
                        break
                    common_prefix += 1
                scores[-1] = scores[-1][max(common_prefix - 1, 0):]

        token_scores = torch.cat(scores)
        if normalization == "mean":
            return float(token_scores.mean().item())
        return float(token_scores.sum().item())

    def _pseudo_log_likelihood(
        self,
        sequence: str,
        normalization: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> float:
        """Compute masked-token pseudo-log-likelihood for one sequence.

        Args:
            sequence: Nucleotide sequence to score.
            normalization: Return the mean or sum over token scores.
            cds: Optional CDS track aligned to sequence.
            splice: Optional splice track aligned to sequence.

        Returns:
            Masked-token pseudo-log-likelihood for the sequence.
        """
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Tokenizer does not define a mask token.")

        prepared_chunks = []
        chunks = self._score_chunks(
            sequence, cds, splice
        )
        if len(chunks) > 1:
            warnings.warn(
                "Pseudo-likelihood is computed independently per chunk; "
                "cross-chunk context is not included.",
                RuntimeWarning,
            )
        for chunk, chunk_cds, chunk_splice in chunks:
            tokenized = self._tokenize_for_logits(
                chunk, chunk_cds, chunk_splice
            )
            input_ids = tokenized["input_ids"].reshape(-1)
            special = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(),
                already_has_special_tokens=True,
            )
            positions = [
                idx for idx, is_special in enumerate(special) if not is_special
            ]
            if not positions:
                raise ValueError(
                    "Pseudo-likelihood requires at least one token per chunk."
                )

            prepared_chunks.append((tokenized, positions))

        scores = []
        for tokenized, positions in prepared_chunks:
            scores.extend(self._masked_token_log_probs(
                tokenized, positions, mask_token_id
            ))

        total = torch.stack(scores)
        if normalization == "mean":
            return float(total.mean().item())
        return float(total.sum().item())

    def _masked_marginal_llr(
        self,
        reference: str,
        alternate: str,
        normalization: str,
        cds: np.ndarray | None,
        splice: np.ndarray | None,
    ) -> float:
        """Score one substitution at every tokenizer position it changes.

        Args:
            reference: Reference allele sequence.
            alternate: Alternate allele sequence.
            normalization: Return the mean or sum over changed token scores.
            cds: Optional CDS track aligned to both sequences.
            splice: Optional splice track aligned to both sequences.

        Returns:
            Reference-minus-alternate masked marginal log-likelihood ratio.
        """
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Tokenizer does not define a mask token.")

        reference_chunks = self._score_chunks(reference, cds, splice)
        alternate_chunks = self._score_chunks(alternate, cds, splice)
        if len(reference_chunks) != len(alternate_chunks):
            raise ValueError(
                "Reference and alternate scoring chunks do not align."
            )

        reference_scores = []
        alternate_scores = []
        for ref_chunk, alt_chunk in zip(reference_chunks, alternate_chunks):
            ref_sequence, ref_cds, ref_splice = ref_chunk
            alt_sequence, alt_cds, alt_splice = alt_chunk
            ref_tokens = self._tokenize_for_logits(
                ref_sequence, ref_cds, ref_splice
            )
            alt_tokens = self._tokenize_for_logits(
                alt_sequence, alt_cds, alt_splice
            )
            ref_ids = ref_tokens["input_ids"].reshape(-1).tolist()
            alt_ids = alt_tokens["input_ids"].reshape(-1).tolist()
            if ref_ids == alt_ids:
                continue
            ref_positions: list[int] = []
            alt_positions: list[int] = []
            for tag, i1, i2, j1, j2 in SequenceMatcher(
                None, ref_ids, alt_ids, autojunk=False
            ).get_opcodes():
                if tag != "equal":
                    ref_positions.extend(range(i1, i2))
                    alt_positions.extend(range(j1, j2))

            ref_special = self.tokenizer.get_special_tokens_mask(
                ref_ids, already_has_special_tokens=True
            )
            alt_special = self.tokenizer.get_special_tokens_mask(
                alt_ids, already_has_special_tokens=True
            )
            ref_positions = [
                idx for idx in ref_positions if not ref_special[idx]
            ]
            alt_positions = [
                idx for idx in alt_positions if not alt_special[idx]
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
        ref_total = torch.stack(reference_scores)
        alt_total = torch.stack(alternate_scores)
        if normalization == "mean":
            return float((ref_total.mean() - alt_total.mean()).item())
        return float((ref_total.sum() - alt_total.sum()).item())

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
                key: value.repeat(
                    batch_size,
                    *([1] * (value.dim() - 1)),
                ).to(self.device)
                for key, value in tokenized.items()
            }
            input_ids = masked["input_ids"].reshape(batch_size, -1)
            row_indices = torch.arange(batch_size, device=self.device)
            position_tensor = torch.tensor(
                batch_positions,
                device=self.device,
            )
            labels = input_ids[row_indices, position_tensor].clone()
            input_ids[row_indices, position_tensor] = mask_token_id
            with torch.inference_mode():
                outputs = self._get_logits_model()(**masked)
                logits = outputs.logits.reshape(
                    batch_size, -1, outputs.logits.shape[-1]
                )[
                    row_indices, position_tensor
                ].float()
                scores.extend(torch.log_softmax(
                    logits, dim=-1
                )[row_indices, labels])
        return scores

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered list of hookable module paths for this model.

        Each path can be passed to model.get_submodule(path). Integer indices
        in extract(layers=...) resolve against this list. String paths must
        name one of these declared hookable layers.

        Discovers the modules named by the ``hookable_layer_patterns`` class
        attribute (a list of fullmatch regexes that each model must declare).
        Override this property entirely if the layers cannot be discovered
        from ``named_modules()``.
        """
        return discover_layers(self.model, self.hookable_layer_patterns)

    def _resolve_layer_paths(
        self,
        layers: list[int | str] | None,
    ) -> list[str]:
        """Convert layers argument to a list of module path strings.

        Args:
            layers: None (all hookable_layers), integers (index into
                hookable_layers, supports negative), or declared path strings.

        Returns:
            Ordered list of module path strings.
        """
        if layers is None:
            return self.hookable_layers
        hookable = self.hookable_layers
        resolved = []
        for layer in layers:
            if isinstance(layer, int):
                resolved.append(hookable[layer])
            else:
                if layer not in hookable:
                    raise ValueError(
                        "Unknown layer {!r}. Valid layers: {}".format(
                            layer, hookable
                        )
                    )
                resolved.append(layer)
        return resolved

    def _register_hooks(
        self,
        layer_paths: list[str],
        detach: bool = True,
    ) -> tuple[list[Any], dict[str, list[torch.Tensor]]]:
        """Register forward hooks on modules at the given paths.

        Args:
            layer_paths: Module paths relative to self.model.
            detach: Detach captured tensors from autograd.

        Returns:
            (handles, activations) where handles must be removed after the
            forward pass and activations[path] accumulates each captured
            output tensor.
        """
        activations: dict[str, list[torch.Tensor]] = {
            p: [] for p in layer_paths
        }
        handles = []
        for path in layer_paths:
            module = self.model.get_submodule(path)

            def make_hook(name: str) -> Any:
                def hook(
                    _module: torch.nn.Module,
                    _input: Any,
                    output: torch.Tensor | tuple[torch.Tensor, ...],
                ) -> None:
                    out = output[0] if isinstance(output, tuple) else output
                    activations[name].append(
                        out.detach() if detach else out
                    )
                return hook

            handles.append(module.register_forward_hook(make_hook(path)))
        return handles, activations

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
        handles, activations = self._register_hooks(
            layer_paths,
            detach=detach,
        )
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

    def _remove_hooks(self, handles: list[Any]) -> None:
        """Remove all registered forward hooks.

        Args:
            handles: List of hook handles returned by _register_hooks.
        """
        for h in handles:
            h.remove()

    def _standard_hf_extract(
        self,
        sequences: list,
        tokenize_fn: Callable[[list], dict[str, torch.Tensor]],
        max_chunk_length: int = 0,
        layers: list[int | str] | None = None,
        return_attentions: bool = False,
        offload_to_cpu: bool = True,
        chunk_fn: Callable[[Any], list] | None = None,
    ) -> tuple[
        dict[str, list[list[torch.Tensor]]],
        dict[str, list[list[torch.Tensor]] | None],
    ]:
        """Run the shared HuggingFace extract() implementation.

        Handles chunking, per-layer extraction, and optional attention weights.
        Hidden states are collected for every requested layer. Attention is
        collected only for layers that actually emit it; a layer that never
        emits attention maps to None in the returned scores dict.

        Args:
            sequences: Pre-processed sequences ready for chunking and
                tokenization. Normally list[str], but may be list[Any] when
                chunk_fn is provided (e.g. list[tuple[str, ndarray]] for
                models that require auxiliary data such as CDS tracks).
            tokenize_fn: Called with a one-element list containing a single
                chunk (the same element type produced by chunk_fn). Returns a
                tokenized dict already moved to the correct device.
            max_chunk_length: Maximum sequence length (characters) per chunk.
                Only used when chunk_fn is None.
            layers: Passed through to _resolve_layer_paths.
            return_attentions: Whether to request attention weights.
            offload_to_cpu: If True, move captured tensors to CPU immediately
                after each chunk to reduce GPU memory pressure.
            chunk_fn: Optional callable that replaces the default
                self.chunk_sequence(seq, max_chunk_length). Receives each
                element of sequences and returns a list of chunks. Use this
                when sequences contain auxiliary data or when a non-standard
                chunking strategy is required.

        Returns:
            (hidden_states, scores) in the standard extract() format.
        """
        if self.attn_implementation != "eager":
            if return_attentions:
                warnings.warn(
                    f"Return attention not available with attn_implementation"
                    f"='{self.attn_implementation}'. To extract attentions, "
                    "initialize the model with attn_implementation='eager'."
                )
            return_attentions = False

        resolved = self._resolve_layer_paths(layers)
        hookable = self.hookable_layers
        layer_to_idx = {
            path: index for index, path in enumerate(hookable)
        }

        # path -> per-sequence -> per-chunk tensors. Pre-allocating a
        # bucket per sequence lets us write directly via the sequence
        # index, avoiding any per-sequence temporaries or a second
        # collation loop.
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
                    outputs = self.model(
                        **toks,
                        output_hidden_states=True,
                        output_attentions=return_attentions,
                    )
                hidden_states = outputs.hidden_states
                attentions = getattr(outputs, "attentions", None)
                for path in resolved:
                    hidden_index = layer_to_idx[path] + 1
                    h = hidden_states[hidden_index][0]
                    hidden_out[path][seq_idx].append(
                        h.cpu() if offload_to_cpu else h
                    )

                    attention_index = hidden_index - 1
                    a = None
                    if return_attentions and attentions is not None:
                        if attention_index < len(attentions):
                            a = attentions[attention_index]
                    if a is not None:
                        a_sq = a[0]  # squeeze batch dim -> (H, T, T)
                        score_acc[path][seq_idx].append(
                            a_sq.cpu() if offload_to_cpu else a_sq
                        )

        # A path is None iff it never produced attention for any chunk.
        score_out = {
            path: (None if all(len(c) == 0 for c in per_seq) else per_seq)
            for path, per_seq in score_acc.items()
        }

        return hidden_out, score_out

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
            sequences: List of sequences to process (DNA bases; T->U conversion
                is done internally per model as needed).
            cds: Optional CDS tracks (used by models that require them).
            splice: Optional splice-site tracks (used by models that require
                them).
            layers: Which layers to extract. None = all hookable_layers.
                Integers index into self.hookable_layers (negative supported).
                Strings are passed as raw get_submodule() paths, allowing
                sub-block access such as 'encoder.layer.3.attention'.
            return_attentions: If True, also extract attention weights.
                For models using Flash Attention or non-transformer layers,
                scores[layer_name] will be None.
            offload_to_cpu: If True (default), move captured tensors to CPU
                immediately after each chunk to reduce GPU memory pressure.

        Returns:
            (hidden_states, scores) where:

            hidden_states: dict mapping layer_name ->
                list[per_sequence [ list[per_chunk [ Tensor(T, D) ]] ]]

            scores: dict mapping layer_name ->
                None  — if not applicable for this layer type (Hyena, Mamba,
                         CNN) or if return_attentions=False
                list[per_sequence [ list[per_chunk [ Tensor(H, T, T) ]] ]]
                        — transformer attention weights

        Raises:
            NotImplementedError: If this model class has not implemented
                extract(). Override in the subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement extract(). "
            "Add a model-specific extract() override."
        )

    def get_peft_target(self) -> "torch.nn.Module":
        """Return the nn.Module that LoRA adapters should be applied to.

        Override in subclasses whose backbone is a non-nn.Module wrapper
        where the trainable nn.Module lives at a different attribute
        (e.g. self.model.model).
        """
        return self.model

    def set_peft_target(self, peft_model: "torch.nn.Module") -> None:
        """Replace the LoRA target module after adapter injection.

        Override alongside get_peft_target() in subclasses that need to
        write the PeftModel back to a non-standard location.

        Args:
            peft_model: PeftModel returned by get_peft_model().
        """
        self.model = peft_model

    def set_inference_mode(self):
        """Set model to inference mode."""
        self.model.eval()
        if hasattr(self, "logits_model"):
            self.logits_model.eval()

    def set_train_mode(self):
        """Set model to training mode."""
        self.model.train()
        if hasattr(self, "logits_model"):
            self.logits_model.train()

    @abstractmethod
    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = mean_pool,
    ) -> list[torch.Tensor]:
        """Embed sequences, optionally using cds/splice tracks.

        Args:
            sequences: List of nucleotide sequences to embed (uses DNA bases).
            cds: List of binary encodings of first nucleotide of each codon.
            splice: List of binary encodings of splice site locations.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embedded sequences with shape (batch_size x H).
        """
        pass

    def chunk_sequence(self, sequence: str, chunk_length: int) -> list[str]:
        """Split sequence into chunks of specified length with given overlap.

        Args:
            sequence: The input string sequence to be chunked.
            chunk_length: The length of each chunk.

        Returns:
            A list of string chunks, where each chunk has the specified length.
        """
        return [
            sequence[i:i + chunk_length]
            for i in range(0, len(sequence), chunk_length)
        ]

    def chunk_tokens(
        self,
        sequence_tokens: list[T],
        chunk_length: int,
    ) -> list[list[T]]:
        """Chunk tokenized sequence into specified length.

        Args:
            sequence_tokens: The tokenized sequence to be chunked.
            chunk_length: The length of each chunk.

        Returns:
            A list of chunked tokens each with specified maximum length.
        """
        chunks = []
        for i in range(0, len(sequence_tokens), chunk_length):
            chunk = sequence_tokens[i:i + chunk_length]
            chunks.append(chunk)

        return chunks

    def _embed_with_chunking(
        self,
        sequences: list[str],
        max_chunk_length: int,
        embed_fn: Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]],
        agg_fn: Callable = mean_pool,
    ) -> list[torch.Tensor]:
        """Embed sequences with chunking and reassembly.

        Handles the common pattern of chunking long sequences, running a
        single forward pass, and reassembling per-sequence embeddings.

        Args:
            sequences: List of sequences to embed.
            max_chunk_length: Maximum chunk length in nucleotides.
            embed_fn: Function that takes a list of sequence chunks and returns
                (hidden_states, pooling_mask) tensors. The pooling_mask
                indicates which tokens to include in aggregation
                (excludes padding and special tokens like CLS/SEP/EOS).
            agg_fn: Function to aggregate token embeddings (default: mean).

        Returns:
            Embeddings with shape (num_sequences, hidden_dim).
        """
        chunks = []
        chunk_counts = []

        for seq in sequences:
            seq_chunks = self.chunk_sequence(seq, max_chunk_length)
            chunks.extend(seq_chunks)
            chunk_counts.append(len(seq_chunks))

        hidden_states, pooling_mask = embed_fn(chunks)

        seq_embeddings = []
        chunk_ptr = 0

        for num_chunks in chunk_counts:
            seq_hidden = hidden_states[chunk_ptr:chunk_ptr + num_chunks]
            seq_mask = pooling_mask[chunk_ptr:chunk_ptr + num_chunks]

            hidden = seq_hidden.reshape(-1, seq_hidden.shape[-1])
            mask = seq_mask.reshape(-1).bool()

            masked_hidden = hidden[mask]
            seq_embeddings.append(agg_fn(masked_hidden))

            chunk_ptr += num_chunks

        return seq_embeddings
