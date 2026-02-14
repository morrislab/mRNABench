from collections.abc import Callable
from functools import partial

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class GENERanno(EmbeddingModel):
    """Inference wrapper for GENERanno.

    GENERanno is a Transformer-encoder genomic foundation model designed for
    metagenomic annotation. It operates at single-nucleotide resolution with
    bidirectional attention over sequences up to 8k base pairs.

    The base checkpoints are pretrained on large-scale DNA corpora (e.g.,
    715B bp prokaryotic and 386B bp eukaryotic variants). Specialized
    "cds-annotator" checkpoints are finetuned for metagenomic CDS calling.

    Link: https://github.com/GenerTeam/GENERanno
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "GENERanno-" + model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize GENERanno inference wrapper.

        Args:
            model_version: Version of GENERanno to load. Valid values are: {
                "prokaryote-0.5b-base",
                "prokaryote-0.5b-cds-annotator",
                "eukaryote-0.5b-base",
                "eukaryote-1.2b-cds-annotator-preview",
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional_dependency to use GENERanno."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "GenerTeam/GENERanno-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModel.from_pretrained(
            "GenerTeam/GENERanno-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(self.device)

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        # Set pad_token to eos_token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = self.model.config
        self.max_length = config.max_position_embeddings

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using GENERanno.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            GENERanno embedding of sequence with shape (1 x H).
            H is:
                1280 for 0.5b models
                2048 for 1.2b models
        """
        chunks = self.chunk_sequence(sequence, self.max_length - 2)

        embedding_chunks = []

        with torch.inference_mode():
            for i, chunk in enumerate(chunks):

                inputs = self.tokenizer(
                    chunk,
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)

                if len(chunks) != 1:
                    if i == 0:
                        inputs = {k: v[:, :-1] for k, v in inputs.items()}
                    elif i == len(chunks) - 1:
                        inputs = {k: v[:, 1:] for k, v in inputs.items()}
                    else:
                        inputs = {k: v[:, 1:-1] for k, v in inputs.items()}

                model_out = self.model(
                    **inputs,
                    output_hidden_states=True
                ).hidden_states[-1]

                embedding_chunks.append(model_out)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for GENERanno.")
