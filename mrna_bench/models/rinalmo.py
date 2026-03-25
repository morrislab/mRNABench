from collections.abc import Callable
from functools import partial

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
            from multimolecule import (
                RnaTokenizer, RiNALMoModel, RiNALMoConfig
            )
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RiNALMo."
            )

        model_path = "multimolecule/{}".format(model_version)
        weights_path = get_model_weights_path()

        self.tokenizer = RnaTokenizer.from_pretrained(
            model_path,
            extra_special_tokens={},
            cache_dir=weights_path
        )

        # Checkpoints are saved as RiNALMoForPreTraining with a 'model.'
        # prefix; RiNALMoModel.from_pretrained doesn't strip it, so we
        # load manually. Issue exists for multimolecule < 0.0.9
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st

        config = RiNALMoConfig.from_pretrained(
            model_path,
            cache_dir=weights_path
        )
        self.model = RiNALMoModel(config)

        ckpt_path = hf_hub_download(
            repo_id=model_path,
            filename="model.safetensors",
            cache_dir=weights_path,
        )
        ckpt = st.load_file(ckpt_path)
        weights = {
            k[len("model."):]: v for k, v in ckpt.items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(weights, strict=False)

        self.model = self.model.to(device)

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
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using RiNALMo.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (hidden_dim,)
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
