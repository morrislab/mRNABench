from collections.abc import Callable
from functools import partial

import torch

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

            from transformers import AutoTokenizer

        except ImportError:
            raise ImportError((
                "Install base_models optional dependency to use "
                "NucleotideTransformerV3."
            ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/NT{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        if self.post_trained:
            self.model = AutoModel.from_pretrained(
                "InstaDeepAI/NT{}".format(model_version),
                trust_remote_code=True,
                cache_dir=get_model_weights_path(),
            ).to(self.device)

            # a species token is required for post-trained models
            self.valid_species = [
                species for species in
                list(self.model.config.species_to_token_id.keys())
                if not species.startswith("<")
            ]
            self.species_id = None
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                "InstaDeepAI/NT{}".format(model_version),
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
            raise ValueError(
                "Setting species is only supported for post-trained "
                "NucleotideTransformerV3 models."
            )

        if species == 'synthetic':
            species = 'human'

            print((
                "Warning: 'synthetic' species not directly supported by "
                "NucleotideTransformerV3 post-trained models. Using 'human' "
                "species token instead."
            ))

        if species not in self.valid_species:
            raise ValueError((
                f"Species '{species}' not valid. Must be one of: "
                f"{self.valid_species}"
            ))

        self.species_id = self.model.encode_species(
            [species]
        ).to(self.device)

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = partial(torch.mean, dim=1)
    ) -> torch.Tensor:
        """Embed sequence using NucleotideTransformerV3.

        Args:
            sequence: Sequence to be embedded.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            NT embedding of sequence with shape (1 x H).
            H is:
                256 for v3_8M_pre
                768 for v3_100M_pre/post
                1536 for v3_650M_pre/post
        """
        if self.post_trained and self.species_id is None:
            raise ValueError((
                "Species must be set for post-trained NucleotideTransformerV3 "
                "models before embedding sequences. Use the `set_species` "
                "method to set the species."
            ))

        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        for _, chunk in enumerate(chunks):

            # NTV3 needs input sequence length to be multiple of 128
            input_ids = self.tokenizer(
                chunk,
                add_special_tokens=False,
                padding=True,
                pad_to_multiple_of=128,
                return_tensors="pt"
            )['input_ids'].to(self.device)

            if self.post_trained:
                model_out = self.model(
                    species_ids=self.species_id,
                    input_ids=input_ids,
                    output_hidden_states=True
                )
            else:
                model_out = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True
                )

            # Remove padding tokens
            model_out = model_out["hidden_states"][-1][:, :len(chunk), :]

            embedding_chunks.append(model_out)

        embedding = torch.cat(embedding_chunks, dim=1)

        aggregate_embedding = agg_fn(embedding)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice, agg_fn):
        """Not supported."""
        raise NotImplementedError("Six track not available for NTV3.")
