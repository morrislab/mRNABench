from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, Protocol, runtime_checkable
import re
import warnings

import numpy as np
import torch


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


def hf_extract(
    hf_model: torch.nn.Module,
    toks: dict[str, torch.Tensor],
    layer_paths: list[str],
    hookable_layers: list[str],
    return_attentions: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
    """Extract hidden states and attention weights from a HuggingFace model.

    Calls the model with output_hidden_states=True and optionally
    output_attentions=True. The hidden_states tuple from HuggingFace
    has index 0 = embedding layer output, and index i+1 = encoder layer i.
    Attention weights may be None when Flash Attention is in use.

    Args:
        hf_model: HuggingFace model supporting output_hidden_states.
        toks: Tokenizer output (input_ids, attention_mask, etc.) on device.
        layer_paths: Subset of hookable_layers paths to extract.
        hookable_layers: Full ordered layer list - used to map paths to
            hidden_states indices (path at position i -> hidden_states[i+1]).
        return_attentions: Whether to request attention weights.

    Returns:
        (hidden_dict, attn_dict) where keys are layer_paths.
        hidden_dict[path]: Tensor(B, T, D)
        attn_dict[path]: Tensor(B, H, T, T) or None
    """
    outputs = hf_model(
        **toks,
        output_hidden_states=True,
        output_attentions=return_attentions,
    )

    hf_hidden = outputs.hidden_states  # tuple of (B, T, D)
    # tuple (B, H, T, T) or None
    hf_attns = getattr(outputs, "attentions", None)

    layer_to_idx = {p: i for i, p in enumerate(hookable_layers)}

    hidden_dict = {}
    attn_dict: dict[str, torch.Tensor | None] = {}

    for path in layer_paths:
        # hidden_states[0] = input_embedding;
        # hookable_layers[i] -> hidden_states[i+1]
        hs_idx = layer_to_idx[path] + 1
        hidden_dict[path] = hf_hidden[hs_idx]

        attn_idx = hs_idx - 1
        attn_dict[path] = None
        if return_attentions and hf_attns is not None:
            if attn_idx < len(hf_attns):
                attn_dict[path] = hf_attns[attn_idx]

    return hidden_dict, attn_dict


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

    default_version: ClassVar[str]
    valid_versions: ClassVar[list[str]]
    default_attn_implementation: ClassVar[str | None]
    valid_attn_implementations: ClassVar[list[str] | None]
    hookable_layer_patterns: ClassVar[list[str]]
    model: torch.nn.Module

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

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered list of hookable module paths for this model.

        Each path can be passed to model.get_submodule(path). Integer indices
        in extract(layers=...) resolve against this list; raw string paths
        are passed through directly (allowing sub-block access such as
        'encoder.layer.3.attention' without pre-registration).

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
                hookable_layers, supports negative), or raw path strings.

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
                resolved.append(layer)
        return resolved

    def _register_hooks(
        self,
        layer_paths: list[str],
    ) -> tuple[list[Any], dict[str, list[torch.Tensor]]]:
        """Register forward hooks on modules at the given paths.

        Args:
            layer_paths: Module paths relative to self.model.

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
                    activations[name].append(out.detach())
                return hook

            handles.append(module.register_forward_hook(make_hook(path)))
        return handles, activations

    def _remove_hooks(self, handles: list[Any]) -> None:
        """Remove all registered forward hooks.

        Args:
            handles: List of hook handles returned by _register_hooks.
        """
        for h in handles:
            h.remove()

    def _compute_layer_scores(
        self,
        layer_name: str,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute attention-equivalent scores for a non-attention layer.

        Override in subclasses to return (H, T, T) proxy scores.

        Planned future implementations:
        - Hyena: per-channel Hyena matrix (HazyResearch/safari issue #45)
        - Mamba: SSM influence matrix from A, B, C, dt parameters

        Args:
            layer_name: Module path of the layer being processed.
            module: The actual nn.Module instance.
            hidden_states: Token representations at this layer, shape (T, D).

        Returns:
            None by default (no scores for this layer type).
        """
        return None

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
                h_dict, a_dict = hf_extract(
                    self.model, toks, resolved, hookable, return_attentions
                )
                for path in resolved:
                    h = h_dict[path][0]  # squeeze batch dim → (T, D)
                    hidden_out[path][seq_idx].append(
                        h.cpu() if offload_to_cpu else h
                    )

                    a = a_dict[path]
                    if a is not None:
                        a_sq = a[0]  # squeeze batch dim → (H, T, T)
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
            sequences: List of sequences to process (DNA bases; T→U conversion
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
        """Set model to inference mode with gradients disabled."""
        self.model.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        """Set model to training mode with gradients enabled."""
        self.model.train()
        torch.set_grad_enabled(True)

    @abstractmethod
    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
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

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> torch.Tensor:
        """Legacy wrapper for embed with a single sequence.

        Args:
            sequence: String of nucleotides to embed (uses DNA bases).
            cds: Binary encoding of first nucleotide of each codon in CDS.
            splice: Binary encoding of splice site locations.
            agg_fn: Method used to aggregate across sequence dimension.

        Returns:
            Embedded sequence with shape (1 x H).
        """
        cds_list = [cds] if cds is not None else None
        splice_list = [splice] if splice is not None else None
        embs = self.embed([sequence], cds_list, splice_list, agg_fn)
        result = torch.stack(embs)
        if result.dim() > 2:
            result = result.squeeze(0)
        return result

    def chunk_sequence(self, sequence: str, chunk_length: int) -> list[str]:
        """Split sequence into chunks of specified length with given overlap.

        Args:
            sequence: The input string sequence to be chunked.
            chunk_length: The length of each chunk.

        Returns:
            A list of string chunks, where each chunk has the specified length.
        """
        chunks = []
        for i in range(0, len(sequence), chunk_length):
            chunk = sequence[i:i + chunk_length]
            chunks.append(chunk)

        return chunks

    def chunk_tokens(
        self,
        sequence_tokens: list[int],
        chunk_length: int,
    ) -> list[list[int]]:
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
        agg_fn: Callable = partial(torch.mean, dim=0),
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
