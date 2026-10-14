from collections.abc import Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from mrna_bench.datasets.dataset_utils import str_to_ohe
from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


class Orthrus(EmbeddingModel):
    """Inference wrapper for Orthrus.

    Orthrus is a RNA foundation model trained using a Mamba backbone. It uses
    a contrastive learning pre-training objective that maximizes similarity
    between RNA splice isoforms and orthologous transcripts. Input length is
    unconstrained due to use of Mamba.

    Link: https://github.com/bowang-lab/Orthrus
    """

    default_version = "orthrus-large-6-track"
    valid_versions = [
        "orthrus-base-4-track",
        "orthrus-large-4-track",
        "orthrus-large-6-track",
    ]
    default_attn_implementation = None
    valid_attn_implementations = None
    hookable_layer_patterns = [r"layers\.\d+"]
    final_layer_pattern = r"norm_f"
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    # mamba_ssm applies in_proj and dt_proj by their weights, bypassing
    # LoRA; x_proj and out_proj are adapted once the fast path is off.
    lora_target_modules = ["x_proj", "out_proj"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("-track", "")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize Orthrus model.

        Args:
            model_version: Version of Orthrus to load. Valid values are: {
                "orthrus-base-4-track",
                "orthrus-large-4-track",
                "orthrus-large-6-track"
            }
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use Orthrus."
            )

        model_hf_path = "quietflamingo/{}".format(model_version)
        model = AutoModel.from_pretrained(
            model_hf_path,
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = model.to(device)

    def tokenize(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """One-hot encode sequences, with cds and splice for 6-track models.

        Args:
            sequences: Nucleotide sequences.
            cds: CDS tracks (required for 6-track models).
            splice: Splice site tracks (required for 6-track models).
            add_special_tokens: Unused; Orthrus has no special tokens.

        Returns:
            "one_hot" model input and its "attention_mask".
        """
        _ = add_special_tokens
        if "6-track" not in self.model_version:
            cds, splice = None, None
        elif cds is None or splice is None:
            raise ValueError(
                "Orthrus 6-track model requires cds and splice tracks."
            )
        one_hot = [
            torch.as_tensor(str_to_ohe(
                sequence,
                cds=None if cds is None else cds[idx],
                splice=None if splice is None else splice[idx],
            ), dtype=torch.float32)
            for idx, sequence in enumerate(sequences)
        ]
        lengths = torch.tensor([len(x) for x in one_hot])
        attention_mask = torch.arange(int(lengths.max())) < lengths[:, None]
        return {
            "one_hot": pad_sequence(one_hot, batch_first=True).to(self.device),
            "attention_mask": attention_mask.long().to(self.device),
        }

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
    ) -> list[torch.Tensor]:
        """Embed sequences using Orthrus.

        Args:
            sequences: List of sequences to embed.
            cds: List of CDS tracks (required for 6-track models).
            splice: List of splice site tracks (required for 6-track models).
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (512,)
        """
        tokens = self.tokenize(sequences, cds, splice)
        hidden_states = self.model.forward(
            tokens["one_hot"], channel_last=True
        )
        return [
            agg_fn(hidden[mask])
            for hidden, mask in zip(hidden_states, self.pooling_mask(tokens))
        ]

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
        """Extract per-layer representations from Orthrus.

        Uses forward hooks on Mamba layers. Scores are None for all layers
        (Mamba SSM, no attention weights). The 6-track model requires cds
        and splice tracks.

        Args:
            sequences: RNA/DNA sequences.
            cds: CDS tracks (required for 6-track model).
            splice: Splice site tracks (required for 6-track model).
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Ignored (no attention scores for Mamba).
            offload_to_cpu: Move tensors to CPU after each sequence.

        Returns:
            (hidden_states, scores); scores are all None.
        """
        resolved = self._resolve_layer_paths(layers)
        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }
        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: None for p in resolved
        }
        for i, seq in enumerate(sequences):
            tokens = self._tokenize_one(
                seq,
                None if cds is None else cds[i],
                None if splice is None else splice[i],
            )
            _, activations = self._run_with_layer_capture(
                resolved,
                lambda: self.model.forward(
                    tokens["one_hot"], channel_last=True
                ),
            )
            for path in resolved:
                h = activations[path][0]
                if h.dim() == 3:
                    h = h[0]  # (T, D)
                hidden_out[path].append([h.cpu() if offload_to_cpu else h])

        return hidden_out, score_out
