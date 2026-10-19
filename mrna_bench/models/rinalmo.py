from collections.abc import Callable

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RiNALMo(EmbeddingModel):
    """Inference wrapper for RiNALMo.

    RiNALMo is a transformer-based RNA foundation model trained on 36M ncRNA
    sequences using MLM and other modern architectural improvements such as
    RoPE, SwiGLU activations, and Flash Attention.

    Link: https://github.com/lbcb-sci/RiNALMo

    This wrapper uses the multimolecule implementation of RiNALMo:
    https://huggingface.co/multimolecule

    Available model versions:
        - rinalmo-giga (0.7B parameters)
        - rinalmo-mega (0.1B parameters)
        - rinalmo-micro (33.5M parameters)
    """

    default_version = "rinalmo-mega"
    valid_versions = ["rinalmo-micro", "rinalmo-mega", "rinalmo-giga"]

    max_length = 8192

    def __init__(self, model_version: str, device: torch.device):
        """Initialize RiNALMo inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions: {
                "rinalmo-giga", "rinalmo-mega", "rinalmo-micro"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from huggingface_hub import hf_hub_download
            from multimolecule import RnaTokenizer, RiNALMoModel
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RiNALMo."
            )

        model_path = "multimolecule/{}".format(model_version)

        # The checkpoint's extra_special_tokens is a list (transformers 5
        # format); transformers 4.57 expects a dict. It only lists <null>.
        self.tokenizer = RnaTokenizer.from_pretrained(
            model_path,
            cache_dir=get_model_weights_path(),
            extra_special_tokens={}
        )

        self.model = RiNALMoModel.from_pretrained(
            model_path,
            cache_dir=get_model_weights_path()
        ).to(device)

        # Checkpoint keys use "model.", not the "rinalmo." 0.0.7 expects.
        checkpoint_file = hf_hub_download(
            model_path,
            "model.safetensors",
            cache_dir=get_model_weights_path(),
        )
        checkpoint = load_file(checkpoint_file)
        weights = {
            key.removeprefix("model."): value
            for key, value in checkpoint.items()
        }
        missing, _ = self.model.load_state_dict(weights, strict=False)
        # Rotary inv_freq is recomputed from config; the pooler is unused.
        missing = [
            key for key in missing
            if not key.endswith("inv_freq") and not key.startswith("pooler.")
        ]
        if missing:
            raise RuntimeError(
                "Parameters absent from checkpoint: {}".format(missing)
            )

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

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

        # Exclude special tokens (CLS at pos 0, EOS at last real pos)
        pooling_mask[:, 0] = 0
        seq_lengths = toks["attention_mask"].sum(dim=1).long()
        for idx in range(pooling_mask.size(0)):
            pooling_mask[idx, seq_lengths[idx] - 1] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Embed sequences using RiNALMo.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embeddings.

        Returns:
            Embeddings with shape (batch_size, hidden_dim).
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]
        effective_max = self.max_length - 2

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=effective_max,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )
