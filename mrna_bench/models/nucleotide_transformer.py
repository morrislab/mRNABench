from collections.abc import Callable

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class NucleotideTransformer(EmbeddingModel):
    """Inference wrapper for NucleotideTransformer.

    NucleotideTransformer is a transformer based DNA foundation model
    pre-trained using MLM on a variety of pre-training datasets ranging from
    the 1000 (human) genomes project to a multi-species dataset, across
    a wide range of parameters. Input is tokenized to 6-mers where possible,
    with a maximum tokenized sequence length of 1000.

    Link: https://github.com/instadeepai/nucleotide-transformer
    """

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "nt-" + model_version

    def __init__(self, model_version: str, device: torch.device):
        """Initialize NucleotideTransformer inference wrapper.

        Args:
            model_version: Version of model to load. Valid versions are: {
                "2.5b-multi-species",
                "2.5b-1000g",
                "500m-human-ref",
                "500m-1000g",
                "v2-50m-multi-species",
                "v2-100m-multi-species",
                "v2-250m-multi-species",
                "v2-500m-multi-species"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        ).to(self.device)

        self.max_length = self.tokenizer.model_max_length

    def embed_sequence(
        self,
        sequence: str,
        overlap: int = 0,
        agg_fn: Callable = torch.mean
    ) -> torch.Tensor:
        """Embed sequence using NucleotideTransformer.

        Args:
            sequence: Sequence to be embedded.
            overlap: Number of tokens overlapping between chunks.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            NT embedding of sequence with shape (1 x H).
        """
        tokenized = self.tokenizer.encode_plus(sequence, verbose=False)

        chunks = self.chunk_tokens(
            tokenized["input_ids"],
            self.max_length,
            overlap
        )

        embedding_chunks = []

        for _, chunk in enumerate(chunks):
            chunk_tt = torch.Tensor(chunk).unsqueeze(0).to(self.device).long()
            attention_mask = torch.ones_like(chunk_tt).to(self.device).long()

            torch_outs = self.model(
                chunk_tt,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True
            )

            model_out = torch_outs["hidden_states"][-1]
            embedding_chunks.append(model_out)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding, dim=1)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, overlap, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for NT.")
