from collections.abc import Callable
import math
import re
from typing import Any
import warnings

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
    TrackOutput,
)
from mrna_bench.datasets.dataset_utils import str_to_ohe
from huggingface_hub import hf_hub_download


class AlphaGenome(EmbeddingModel):
    """Inference wrapper for AlphaGenome.

    AlphaGenome is a deep learning model for predicting functional genomic
    activity from DNA sequence. It uses a transformer-based architecture
    built on convolutional layers and self-attention mechanisms to model
    long-range interactions in the genome. AlphaGenome is trained on 200 kb
    genomic windows and predicts a range of functional readouts across
    multiple human and mouse datasets. Here we use the PyTorch
    implementation of AlphaGenome from GenomicsXAI, which is based on the
    original AlphaGenome model from DeepMind.

    Link: https://github.com/google-deepmind/alphagenome_research
    Link: https://github.com/genomicsxai/alphagenome-pytorch
    """

    default_version = "alphagenome"
    valid_versions = ["alphagenome"]
    default_attn_implementation = "eager"
    valid_attn_implementations = ["eager"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.TRACKS,
    })
    # encoder down blocks (0-5) -> transformer block MLPs (tower.blocks.N.mlp,
    # the last op in each block, since the block container is a ModuleDict with
    # no forward) -> decoder up blocks (0-6).
    hookable_layer_patterns = [
        r"encoder\.down_blocks\.\d+",
        r"tower\.blocks\.\d+\.mlp",
        r"decoder\.up_blocks\.\d+",
    ]

    lora_target_modules = ["q_proj", "k_proj", "v_proj", "linear_embedding"]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize AlphaGenome.

        Args:
            model_version: Version of model used. Valid versions: {
                "alphagenome"
            }
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )
        self.max_length = 1_048_576  # up to 1048576 bases

        try:
            from alphagenome_pytorch import AlphaGenome
        except ImportError:
            raise ImportError("AlphaGenome missing required dependencies.")

        model_weights_path = hf_hub_download(
            repo_id='gtca/alphagenome_pytorch',
            filename='model_all_folds.safetensors',
            cache_dir=get_model_weights_path(),
        )

        self.model = AlphaGenome.from_pretrained(
            model_weights_path,
            device=device,
        )

        self.species = 0  # default: human

    def set_species(self, species: str) -> None:
        """Set species for AlphaGenome.

        Args:
            species: Species name. Must be "human" or "mouse".
                "synthetic" is accepted but maps to "human".
        """
        if species == "synthetic":
            warnings.warn(
                "Warning: 'synthetic' sequences not directly supported by "
                "AlphaGenome. Using 'human' organism index instead."
            )
            self.species = 0
            return

        if species not in ["human", "mouse"]:
            warnings.warn(
                f"Warning: Unrecognized species '{species}' for AlphaGenome. "
                "Defaulting to 'human' organism index."
            )
            self.species = 0
            return

        self.species = 0 if species == "human" else 1

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using AlphaGenome.

        Chunks with the same natural 2048-base padded length share a model
        call so padding does not change the representation contract.

        Args:
            sequences: List of nucleotide sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Aggregation function applied along the length dimension.

        Returns:
            List of embeddings; default shape per sequence: (1536,).
        """
        _, _ = cds, splice
        if not sequences:
            return []

        buckets: dict[int, list[tuple[int, int, str]]] = {}
        for sequence_idx, sequence in enumerate(sequences):
            for chunk_idx, chunk in enumerate(
                self.chunk_sequence(sequence, self.max_length)
            ):
                padded_length = math.ceil(len(chunk) / 2048) * 2048
                buckets.setdefault(padded_length, []).append(
                    (sequence_idx, chunk_idx, chunk)
                )

        chunks_by_sequence: list[list[tuple[int, torch.Tensor]]] = [
            [] for _ in sequences
        ]
        model: Any = self.model
        for padded_length, records in buckets.items():
            for batch_records in self.chunk_tokens(
                records, max(1, len(sequences))
            ):
                batch = torch.stack([
                    torch.tensor(
                        str_to_ohe(
                            chunk + "N" * (padded_length - len(chunk))
                        ),
                        dtype=torch.float32,
                    )
                    for _, _, chunk in batch_records
                ]).to(self.device)
                organism_index = torch.full(
                    (len(batch_records),),
                    self.species,
                    dtype=torch.long,
                    device=self.device,
                )
                # cuDNN TF32 changes convolution results with batch shape.
                # Use full FP32 so persisted embeddings are batch-size stable.
                with self._full_precision_cudnn():
                    embeddings = model.encode(
                        batch, organism_index, resolutions=(1,)
                    )["embeddings_1bp"]

                for batch_idx, (
                    sequence_idx, chunk_idx, chunk
                ) in enumerate(batch_records):
                    chunks_by_sequence[sequence_idx].append(
                        (chunk_idx, embeddings[batch_idx, :len(chunk)])
                    )

        return [
            agg_fn(torch.cat([
                chunk for _, chunk in sorted(chunks, key=lambda item: item[0])
            ], dim=0))
            for chunks in chunks_by_sequence
        ]

    def predict_tracks(
        self,
        sequences: list[str],
        heads: tuple[str, ...] | None = None,
        resolution: int = 1,
    ) -> list[TrackOutput]:
        """Return AlphaGenome's native genomic tracks.

        Args:
            sequences: Genomic sequences to predict.
            heads: Optional output heads to return. Defaults to every head
                supporting the requested resolution.
            resolution: Output resolution in nucleotides.

        Returns:
            Model-native tracks aligned to each input sequence.
        """
        if resolution not in {1, 128}:
            raise ValueError("resolution must be 1 or 128.")
        if any(len(sequence) > self.max_length for sequence in sequences):
            raise ValueError(
                "predict_tracks accepts one AlphaGenome window per sequence."
            )

        model: Any = self.model
        selected_heads = heads or tuple(
            name for name, head in model.heads.items()
            if resolution in head.resolutions
        )
        outputs = []
        for sequence in sequences:
            padded_length = math.ceil(len(sequence) / 2048) * 2048
            padded = sequence + "N" * (padded_length - len(sequence))
            batch = torch.tensor(
                str_to_ohe(padded),
                dtype=torch.float32,
            ).unsqueeze(0).to(self.device)
            organism = torch.tensor(
                [self.species], dtype=torch.long, device=self.device
            )
            predictions = model.predict(
                batch,
                organism,
                resolutions=(resolution,),
                heads=selected_heads,
            )
            length = math.ceil(len(sequence) / resolution)
            values = {
                head: predictions[head][resolution][0, :length]
                for head in selected_heads
            }
            outputs.append(TrackOutput(values, 0, resolution))
        return outputs

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
        """Extract per-layer hidden states from AlphaGenome.

        Uses forward hooks on named modules. Layer paths follow the pattern:
        - encoder.down_blocks.{0-5} : encoder hidden states (NCL, downsampled)
        - tower.blocks.{0-8}.mlp : transformer MLP output delta (NLC)
        - decoder.up_blocks.{0-6} : decoder hidden states (NCL, upsampled)

        Note: transformer blocks are hooked at the MLP sub-module. Because
        each block's container is a ModuleDict with no forward method, the
        residual addition (x = x + mlp(x)) is a bare tensor op that hooks
        can't intercept directly. Instead, the hook captures input[0] + output
        from the MLP module, which reconstructs the full post-residual state.

        Attention weights cannot be extracted from AlphaGenome without
        modifying the library (attn_weights is a local variable in MHABlock).
        scores will always be None regardless of return_attentions.

        Args:
            sequences: DNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Ignored; attention is not hookable.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores) where scores[path] is always None.
        """
        _, _ = cds, splice

        if return_attentions:
            warnings.warn(
                "AlphaGenome attention weights are not extractable via hooks "
                "(attn_weights is a local variable in MHABlock). "
                "scores will be None for all layers."
            )

        resolved = self._resolve_layer_paths(layers)
        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }
        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: None for p in resolved
        }

        # Transformer MLP paths need input[0] + output to get the post-residual
        # block state. All other paths use the standard output-only hook.
        _mlp_re = re.compile(r"tower\.blocks\.\d+\.mlp")
        mlp_paths = [p for p in resolved if _mlp_re.fullmatch(p)]
        other_paths = [p for p in resolved if not _mlp_re.fullmatch(p)]

        def pad_to_multiple(seq: str, multiple: int = 2048) -> str:
            target = math.ceil(len(seq) / multiple) * multiple
            return seq + "N" * (target - len(seq))

        for seq in sequences:
            chunks = self.chunk_sequence(seq, self.max_length)
            seq_hidden: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved
            }

            for chunk in chunks:
                padded_chunk = pad_to_multiple(chunk)
                batch = torch.tensor(
                    str_to_ohe(padded_chunk),
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                handles, activations = self._register_hooks(other_paths)

                # For transformer MLP layers: hook captures input[0] + output
                # to reconstruct the post-residual block state.
                mlp_activations: dict[str, list[torch.Tensor]] = {
                    p: [] for p in mlp_paths
                }
                mlp_handles = []
                for path in mlp_paths:
                    module = self.model.get_submodule(path)

                    def make_residual_hook(name: str) -> Any:
                        def hook(
                            _mod: torch.nn.Module,
                            inp: tuple,
                            out: torch.Tensor,
                        ) -> None:
                            mlp_activations[name].append(
                                (inp[0] + out).detach()
                            )
                        return hook

                    mlp_handles.append(
                        module.register_forward_hook(make_residual_hook(path))
                    )

                organism_index = torch.tensor(
                    [self.species], dtype=torch.long, device=self.device
                )
                try:
                    model: Any = self.model
                    model.encode(batch, organism_index)
                finally:
                    self._remove_hooks(handles)
                    self._remove_hooks(mlp_handles)

                for path in other_paths:
                    h = activations[path][0]
                    if h.dim() == 3:
                        h = h[0]
                    seq_hidden[path].append(h.cpu() if offload_to_cpu else h)

                for path in mlp_paths:
                    h = mlp_activations[path][0]
                    if h.dim() == 3:
                        h = h[0]
                    seq_hidden[path].append(h.cpu() if offload_to_cpu else h)

            for path in resolved:
                hidden_out[path].append(seq_hidden[path])

        return hidden_out, score_out
