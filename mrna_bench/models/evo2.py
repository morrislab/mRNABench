from collections.abc import Callable
import warnings

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel


def fp8_available(device: torch.device) -> bool:
    """Check whether FP8 input projections can run on the device.

    Args:
        device: PyTorch device the model runs on.

    Returns:
        True if TransformerEngine is installed and the GPU supports FP8
        (compute capability 8.9 or higher).
    """
    if device.type != "cuda":
        return False

    try:
        import transformer_engine.pytorch  # noqa: F401
    except ImportError:
        return False

    return torch.cuda.get_device_capability(device) >= (8, 9)


class Evo2(EmbeddingModel):
    """Inference wrapper for Evo2.

    Evo2 is a StripedHyena2-based DNA foundation model trained on the
    OpenGenome2 dataset using an autoregressive scheme at single nucleotide
    resolution. Owing to its StripedHyena2 backbone, it has an ultra long
    context window. The `base` variants can handle sequences up to 8192
    nucleotides in length while the larger variants can handle sequences up
    to 1 million nucleotides in length.

    Checkpoints are loaded from HuggingFace ports of the Arc Institute
    releases, which bundle their own model code.

    Link: https://github.com/ArcInstitute/evo2
    """

    default_version = "evo2_7b"
    valid_versions = [
        "evo2_40b",
        "evo2_20b",
        "evo2_7b",
        "evo2_7b_262k",
        "evo2_40b_base",
        "evo2_7b_base",
        "evo2_1b_base",
    ]

    # Ported repo and pinned revision for each version.
    version_to_repo = {
        "evo2_40b": (
            "Taykhoom/Evo2-40B-1M",
            "9d14ad59e9808b98142c178909139314b2d63c2e",
        ),
        "evo2_20b": (
            "Taykhoom/Evo2-20B-1M",
            "d4b70970a14ce40384fb60f0e914840e8eacd5c9",
        ),
        "evo2_7b": (
            "Taykhoom/Evo2-7B-1M",
            "98c6c51ed7f2a0b8f7af47c1939a4db3cc093a6f",
        ),
        "evo2_7b_262k": (
            "Taykhoom/Evo2-7B-262K",
            "fdea0b5a4b9e81ce902b8ac4c4989ced03dc3f65",
        ),
        "evo2_40b_base": (
            "Taykhoom/Evo2-40B-8K",
            "a8738015ecff63ea951de4cad98b04fa0ec1dda1",
        ),
        "evo2_7b_base": (
            "Taykhoom/Evo2-7B-8K",
            "56efa143b72aaf6440a32d4face2a9a21a136666",
        ),
        "evo2_1b_base": (
            "Taykhoom/Evo2-1B-8K",
            "bd8075a818d6dedf9ea6c04416283f39d2bac21c",
        ),
    }

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Evo2.

        Args:
            model_version: Version of model used. Valid versions: {
                "evo2_40b",
                "evo2_20b",
                "evo2_7b",
                "evo2_7b_262k",
                "evo2_40b_base",
                "evo2_7b_base",
                "evo2_1b_base",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use Evo2."
            )

        repo_id, revision = self.version_to_repo[model_version]
        hub_kwargs = {
            "revision": revision,
            "cache_dir": get_model_weights_path(),
            "trust_remote_code": True,
        }

        config = AutoConfig.from_pretrained(repo_id, **hub_kwargs)
        if config.use_fp8_input_projections and not fp8_available(device):
            warnings.warn(
                "FP8 input projections need TransformerEngine and a GPU with "
                "compute capability 8.9+; running them in bf16 instead. "
                "Embeddings will differ from the FP8 reference model."
            )
            config.use_fp8_input_projections = False

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **hub_kwargs)
        self.model = AutoModel.from_pretrained(
            repo_id,
            config=config,
            attn_implementation="flash_attention_2",
            **hub_kwargs,
        ).to(device)
        self.max_length = self.tokenizer.model_max_length

        # Embed with the middle block's pre-norm output and the final norm.
        self.middle_layer_idx = config.num_layers // 2
        self.middle_block = self.model.backbone.blocks[self.middle_layer_idx]

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed a single sequence using Evo2.

        Args:
            sequence: Sequence to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embedding with shape (1, 2 * hidden_dim).
        """
        _, _ = cds, splice

        chunks = self.chunk_sequence(sequence, self.max_length)
        embedding_chunks = []

        for chunk in chunks:
            toks = self.tokenizer([chunk], return_tensors="pt").to(self.device)
            outputs = self.model(**toks, output_hidden_states=True)

            # hidden_states[i] is the input to block i.
            middle = self.middle_block.pre_norm(
                outputs.hidden_states[self.middle_layer_idx]
            )
            embedding_chunks.append(
                torch.cat([middle, outputs.last_hidden_state], dim=-1)[0]
            )

        return agg_fn(
            torch.cat(embedding_chunks, dim=0), dim=0
        ).float().unsqueeze(0)

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using Evo2.

        Processes sequences one at a time: StripedHyena's convolutions don't
        fully isolate sequences in a padded batch.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Evo2 embeddings with shape (batch_size, 2 * hidden_dim).
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            all_embeddings.append(self.embed_sequence(sequence, agg_fn=agg_fn))

        return torch.cat(all_embeddings, dim=0)
