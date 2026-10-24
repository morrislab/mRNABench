from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


class UTRLM(EmbeddingModel):
    """Inference wrapper for UTR-LM.

    UTR-LM is a transformer-based mRNA foundation model that is pre-trained on
    random and endogenous 5'UTR sequences from various species using MLM.

    Link: https://github.com/a96123155/UTR-LM
    """

    default_version = "UTR-LM-MLMSI"
    valid_versions = [
        "UTR-LM-MLMSI",
        "UTR-LM-MLMSISS",
        "UTR-LM-MLM",
        "UTR-LM-MLMSS",
    ]
    default_attn_implementation = "sdpa"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]
    final_layer_pattern = r"emb_layer_norm_after"
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
    ]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "UTR-LM-MLMSI": "utrlm-te-el",
            "UTR-LM-MLMSISS": "utrlm-mrl",
            "UTR-LM-MLM": "utrlm-base",
            "UTR-LM-MLMSS": "utrlm-ss",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize UTR-LM inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions: {
                "UTR-LM-MLMSI", "UTR-LM-MLMSISS", "UTR-LM-MLM",
                "UTR-LM-MLMSS"
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
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install base_models optional dependency to use UTR-LM."
            ) from exc

        hub_id = "Taykhoom/{}".format(model_version)
        weights_path = get_model_weights_path()

        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=weights_path
        )

        dtype = self._get_inference_dtype()

        language_model = AutoModelForMaskedLM.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=weights_path,
            attn_implementation=self.attn_implementation,
            dtype=dtype,
        ).to(device)
        self._set_logits_model(language_model)
        self.max_length = self.tokenizer.model_max_length
        self.sequence_score_chunk_length = self.max_length - 2

    def _tokenizer_text(self, sequence: str) -> str:
        """Write RNA bases in the DNA alphabet the UTR-LM tokenizer reads.

        Args:
            sequence: Nucleotide sequence.

        Returns:
            Sequence in DNA bases.
        """
        return sequence.replace("U", "T")

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool()
    ) -> list[torch.Tensor]:
        """Embed sequences using UTR-LM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (128,)
        """
        _, _ = cds, splice
        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
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
        """Extract per-layer representations from UTR-LM.

        Args:
            sequences: RNA/DNA sequences (T or U bases; U->T applied internally
                to match the UTR-LM port's DNA-alphabet tokenizer).
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice
        return self._standard_hf_extract(
            sequences=sequences,
            max_chunk_length=self.max_length - 2,
            layers=layers,
            return_attentions=return_attentions,
            offload_to_cpu=offload_to_cpu,
        )
