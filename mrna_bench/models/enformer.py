from collections.abc import Callable
import math

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
    TrackOutput,
)
from mrna_bench.datasets.dataset_utils import str_to_ohe


class Enformer(EmbeddingModel):
    """Inference wrapper for Enformer.

    Enformer is a deep learning model for predicting functional genomic
    activity from DNA sequence. It uses a transformer-based architecture
    built on convolutional layers and self-attention mechanisms to model
    long-range interactions in the genome. Enformer is trained on 200 kb
    genomic windows and predicts a range of functional readouts across
    multiple human and mouse datasets. Here we use the PyTorch
    implementation of Enformer from EleutherAI, which is based on the
    original Enformer model from DeepMind.

    Link: https://github.com/google-deepmind/deepmind-research
        (Under enformer directory)
    Link: https://github.com/lucidrains/enformer-pytorch
    """

    default_version = "enformer-official-rough"
    valid_versions = ["enformer-official-rough"]
    default_attn_implementation = "eager"
    valid_attn_implementations = ["eager"]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.TRACKS,
    })
    # stem (1) -> conv_tower blocks (6) -> transformer blocks (11)
    hookable_layer_patterns = [
        r"stem",
        r"conv_tower\.\d+",
        r"transformer\.\d+",
    ]

    lora_target_modules = ["to_q", "to_k", "to_v", "to_out"]

    prediction_window = 114_688  # embedding is of the center 114688 bases
    bin_size = 128  # embedding is in 128 base bins

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-official-rough", "")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Enformer.

        Args:
            model_version: Version of model used. Valid versions: {
                "enformer-official-rough"
            }
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )
        self.max_length = 196_608  # up to 196608 bases

        try:
            from enformer_pytorch import from_pretrained
        except ImportError:
            raise ImportError("Enformer missing required dependencies.")

        self.model = from_pretrained(
            f'EleutherAI/{model_version}',
            cache_dir=get_model_weights_path()
        ).to(device)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using Enformer in one fixed-length model batch.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (3072,)
        """
        _, _ = cds, splice
        if not sequences:
            return []

        records = []
        for sequence_idx, sequence in enumerate(sequences):
            for chunk_idx, chunk in enumerate(
                self.chunk_sequence(sequence, self.max_length)
            ):
                padding_total = self.max_length - len(chunk)
                padding_left = (
                    padding_total // 2 // self.bin_size * self.bin_size
                )
                padding_right = self.max_length - len(chunk) - padding_left
                padded = "N" * padding_left + chunk + "N" * padding_right
                records.append(
                    (sequence_idx, chunk_idx, chunk, padding_left, padded)
                )

        chunks_by_sequence: list[list[tuple[int, torch.Tensor]]] = [
            [] for _ in sequences
        ]
        for batch_records in self.chunk_tokens(
            records, max(1, len(sequences))
        ):
            batch = torch.stack([
                torch.tensor(str_to_ohe(padded), dtype=torch.float32)
                for _, _, _, _, padded in batch_records
            ]).to(self.device)
            _, embedded = self.model(
                batch,
                return_embeddings=True,
                target_length=-1,
            )
            for batch_idx, (
                sequence_idx, chunk_idx, chunk, padding_left, _
            ) in enumerate(batch_records):
                start_bin = padding_left // self.bin_size
                end_bin = math.ceil(
                    (padding_left + len(chunk)) / self.bin_size
                )
                chunks_by_sequence[sequence_idx].append(
                    (chunk_idx, embedded[batch_idx, start_bin:end_bin])
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
        species: str = "human",
    ) -> list[TrackOutput]:
        """Return Enformer's native human or mouse tracks.

        Args:
            sequences: Genomic sequences to predict.
            species: Species-specific prediction head to use.

        Returns:
            Model-native tracks aligned to each input sequence.
        """
        if species not in {"human", "mouse"}:
            raise ValueError("species must be 'human' or 'mouse'.")
        if any(len(sequence) > self.max_length for sequence in sequences):
            raise ValueError(
                "predict_tracks accepts one Enformer window per sequence."
            )

        outputs = []
        for sequence in sequences:
            padding_total = self.max_length - len(sequence)
            padding_left = (
                padding_total // 2 // self.bin_size * self.bin_size
            )
            padding_right = self.max_length - len(sequence) - padding_left
            padded = (
                "N" * padding_left + sequence + "N" * padding_right
            )
            batch = torch.tensor(
                str_to_ohe(padded),
                dtype=torch.float32,
            ).unsqueeze(0).to(self.device)
            predictions = self.model(
                batch,
                head=species,
                target_length=-1,
            )
            start = padding_left // self.bin_size
            end = math.ceil(
                (padding_left + len(sequence)) / self.bin_size
            )
            outputs.append(TrackOutput(
                {species: predictions[0, start:end]},
                0,
                self.bin_size,
            ))
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
        """Extract per-layer representations from Enformer.

        Uses forward hooks. CNN/stem layers (stem, conv_tower.*) produce
        representations with reduced sequence length due to downsampling.
        Transformer layers (transformer.*) output (T, D) directly.

        Attention weights are available for transformer layers when
        return_attentions=True. An input hook on the attn_dropout submodule
        captures the pre-dropout (B, H, T, T) softmax attention matrix.
        CNN/stem layers always return None for scores.

        Args:
            sequences: DNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: If True, transformer layers return
                (H, T, T) attention tensors.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); scores[path] is None for CNN/stem layers
            and list[list[Tensor(H,T,T)]] for transformer layers.
        """
        _, _ = cds, splice

        resolved = self._resolve_layer_paths(layers)
        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }

        # Eligible paths: top-level transformer blocks
        # (all use eager attention)
        attn_paths = set()
        if return_attentions:
            for p in resolved:
                if p.startswith("transformer.") and p.count(".") == 1:
                    attn_paths.add(p)

        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: ([] if p in attn_paths else None) for p in resolved
        }

        def center_padding(seq: str, length: int) -> tuple[str, int]:
            centered = (length - len(seq)) // 2
            padding_left = centered // self.bin_size * self.bin_size
            padding_right = length - len(seq) - padding_left
            return "N" * padding_left + seq + "N" * padding_right, padding_left

        for seq in sequences:
            chunks = self.chunk_sequence(seq, self.max_length)
            seq_hidden: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved
            }
            seq_score: dict[str, list[torch.Tensor]] = {
                p: [] for p in attn_paths
            }

            for chunk in chunks:
                padded_chunk, padding_left = center_padding(
                    chunk, self.max_length
                )
                batch = torch.tensor(
                    str_to_ohe(padded_chunk),
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                handles, activations = self._register_hooks(resolved)

                # Register input hooks on attn_dropout for attention capture
                score_store: dict[str, list[torch.Tensor]] = {
                    p: [] for p in attn_paths
                }
                attn_handles = []
                for path in attn_paths:
                    attn_drop_mod = self.model.get_submodule(
                        path + ".0.fn.1.attn_dropout"
                    )

                    def make_attn_hook(name: str):
                        def attn_hook(
                            _: torch.nn.Module,
                            inp: tuple,
                            __: torch.Tensor,
                        ) -> None:
                            score_store[name].append(inp[0].detach())
                        return attn_hook

                    attn_handles.append(
                        attn_drop_mod.register_forward_hook(
                            make_attn_hook(path)
                        )
                    )

                try:
                    self.model(batch, return_embeddings=True, target_length=-1)
                finally:
                    self._remove_hooks(handles)
                    for handle in attn_handles:
                        handle.remove()

                for path in resolved:
                    h = activations[path][0]
                    if h.dim() == 3:
                        h = h[0]  # (T, D)
                    seq_hidden[path].append(h.cpu() if offload_to_cpu else h)

                for path in attn_paths:
                    if score_store[path]:
                        w = score_store[path][0][0]  # (1,H,T,T) -> (H,T,T)
                        seq_score[path].append(
                            w.cpu() if offload_to_cpu else w)

            for path in resolved:
                hidden_out[path].append(seq_hidden[path])
            for path in attn_paths:
                score_list = score_out[path]
                assert isinstance(score_list, list)
                score_list.append(seq_score[path])

        return hidden_out, score_out
