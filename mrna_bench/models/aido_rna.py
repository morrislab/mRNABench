from collections.abc import Callable

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel, ModelBehavior
from mrna_bench.models.pooling import MeanPool


class AIDORNA(EmbeddingModel):
    """Inference wrapper for AIDO.RNA.

    AIDO.RNA is a transformer-based RNA foundation model. It is trained using
    masked language modelling on 42 million non-coding RNA sequences, with
    domain adaptation models available for protein coding sequences.

    Link: https://github.com/genbio-ai/ModelGenerator
    """

    default_version = "AIDO.RNA-650M-CDS"
    valid_versions = [
        "AIDO.RNA-650M",
        "AIDO.RNA-650M-CDS",
        "AIDO.RNA-1.6B",
        "AIDO.RNA-1.6B-CDS",
        "AIDO.RNA-1M-MARS",
        "AIDO.RNA-25M-MARS",
        "AIDO.RNA-300M-MARS",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "sdpa",
        "flash_attention_2",
    ]
    hookable_layer_patterns = [r"encoder\.layer\.\d+"]
    final_layer_pattern = r"encoder\.ln"
    uses_rna_alphabet = False
    lora_target_modules = [
        "query", "key", "value", "dense",
        "gate_proj", "up_proj", "down_proj",
    ]
    supported_behaviors = frozenset({
        ModelBehavior.EMBEDDING,
        ModelBehavior.PSEUDO_LIKELIHOOD,
    })

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "AIDO.RNA-650M": "aido-rna-650m",
            "AIDO.RNA-650M-CDS": "aido-rna-650m-cds",
            "AIDO.RNA-1.6B": "aido-rna-1b600m",
            "AIDO.RNA-1.6B-CDS": "aido-rna-1b600m-cds",
            "AIDO.RNA-1M-MARS": "aido-rna-mars-1m",
            "AIDO.RNA-25M-MARS": "aido-rna-mars-25m",
            "AIDO.RNA-300M-MARS": "aido-rna-mars-300m",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize AIDO.RNA.

        Args:
            model_version: Version of model used. Valid versions: {
                "AIDO.RNA-650M",
                "AIDO.RNA-650M-CDS",
                "AIDO.RNA-1.6B",
                "AIDO.RNA-1.6B-CDS",
                "AIDO.RNA-1M-MARS",
                "AIDO.RNA-25M-MARS",
                "AIDO.RNA-300M-MARS",
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
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use AIDO.RNA."
            )

        hub_id = "Taykhoom/{}".format(model_version)

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
        self.sequence_score_chunk_length = self.max_length - 2

    def _tokenizer_text(self, sequence: str) -> str:
        """Write RNA bases in the DNA alphabet the AIDO.RNA tokenizer reads.

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
        """Embed sequences using AIDO.RNA.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1280,) for 650M model
            - default (mean): (2048,) for 1.6B model
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
        """Extract per-layer representations from AIDO.RNA.

        Args:
            sequences: RNA sequences.
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
