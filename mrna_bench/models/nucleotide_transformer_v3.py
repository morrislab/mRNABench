from collections.abc import Callable
from typing import Optional
import warnings
from functools import partial

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class NucleotideTransformerV3(EmbeddingModel):
    """Inference wrapper for NucleotideTransformer.

    NucleotideTransformerV3 is a Transformer/CNN-based DNA foundation model
    trained on the OpenGenome2 dataset using a masked language modeling
    objective at single nucleotide resolution. Owing to its U-Net style
    architecture that downsamples (convolutions), processes with a
    Transformer tower at the bottleneck, and then upsamples (convolutions),
    it can handle ultra long sequences, and is trained on sequences up to
    1 MB. While it can in principle handle sequences longer than 1 MB due
    to its use of RoPE positional embeddings, due to GPU memory constraints,
    we limit the maximum sequence length to 1,000,064 nucleotides. This can
    be increased if more GPU memory is available. The post-trained versions
    of the model were further trained to predict 16K+ genomic tracks, but
    here only the embedding capabilities are used.

    Link: https://github.com/instadeepai/nucleotide-transformer
    """

    max_length = 1_000_064
    default_version = "v3_650M_post"
    valid_versions = [
        "v3_8M_pre",
        "v3_100M_pre",
        "v3_650M_pre",
        "v3_100M_post",
        "v3_650M_post",
    ]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "nt_" + model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NucleotideTransformer inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions are: {
                "v3_8M_pre",
                "v3_100M_pre",
                "v3_650M_pre",
                "v3_100M_post",
                "v3_650M_post"

            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        self.post_trained = "post" in model_version

        try:
            if self.post_trained:
                from transformers import AutoModel
            else:
                from transformers import AutoModelForMaskedLM

            from transformers import AutoTokenizer, AutoConfig

        except ImportError:
            raise ImportError((
                "Install base_models optional dependency to use "
                "NucleotideTransformerV3."
            ))

        self.config = AutoConfig.from_pretrained(
            "InstaDeepAI/NT{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/NT{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        if self.post_trained:
            self.model = AutoModel.from_pretrained(
                "InstaDeepAI/NT{}".format(model_version),
                config=self.config,
                trust_remote_code=True,
                cache_dir=get_model_weights_path(),
            ).to(self.device)

            s_ids = self.config.species_to_token_id.keys()

            # a species token is required for post-trained models
            self.valid_species = [
                species for species in s_ids
                if not species.startswith("<")
            ]
            self.species_id: Optional[torch.Tensor] = None
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                "InstaDeepAI/NT{}".format(model_version),
                config=self.config,
                trust_remote_code=True,
                cache_dir=get_model_weights_path(),
            ).to(self.device)

    def set_species(self, species: str):
        """Set species for post-trained NucleotideTransformerV3 model.

        Args:
            species: Species name. Must be one of the valid species names
                supported by the model.
        """
        if not self.post_trained:
            warnings.warn((
                "Setting species is only supported for post-trained "
                "NucleotideTransformerV3 models. Ignoring species input."
            ))
            return

        if species == "synthetic":
            species = 'human'

            warnings.warn((
                "Warning: 'synthetic' species not directly supported by "
                "NucleotideTransformerV3 post-trained models. Using 'human' "
                "species token instead."
            ))

        if species not in self.valid_species:
            raise ValueError((
                f"Species '{species}' not valid. Must be one of: "
                f"{self.valid_species}"
            ))

        self.species_id = self.model.encode_species(  # type: ignore[operator]
            [species]
        ).to(self.device)

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding tokens. NTV3 does not use special tokens like CLS/SEP.
        """
        toks = self.tokenizer(
            chunks,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
            return_special_tokens_mask=True
        ).to(self.device)

        if self.post_trained:

            assert self.species_id is not None

            s_ids = self.species_id.expand(
                len(chunks)
            )

            hidden_states = self.model(
                species_ids=s_ids,
                input_ids=toks["input_ids"],
                output_hidden_states=True
            ).hidden_states[-1]
        else:
            hidden_states = self.model(
                input_ids=toks["input_ids"],
                output_hidden_states=True
            ).hidden_states[-1]

        pooling_mask = 1 - toks["special_tokens_mask"]

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using NucleotideTransformerV3.

        Each sequence is processed in its own forward pass to avoid the
        U-Net's convolutional avg_pool layers producing context-dependent
        outputs when sequences of different lengths are padded together.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (256,) for `v3_8M_pre`
            - default (mean): (768,) for `v3_100M_pre/post`
            - default (mean): (1536,) for `v3_650M_pre/post`
        """
        _, _ = cds, splice

        if self.post_trained and self.species_id is None:

            self.set_species("human")

            warnings.warn((
                "Species must be set for post-trained NucleotideTransformerV3 "
                "models. Using default species_id for embedding ('human')."
                " Use the `set_species` method to change the species."
            ))

        return [
            self._embed_with_chunking(
                sequences=[seq],
                max_chunk_length=self.max_length,
                embed_fn=self._forward_chunks,
                agg_fn=agg_fn,
            )[0]
            for seq in sequences
        ]
