from collections.abc import Callable

import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.datasets.dataset_utils import str_to_ohe

class Borzoi(EmbeddingModel):
    """Inference wrapper for Borzoi.

    Borzoi is a deep learning model for predicting RNA-seq coverage
    from DNA sequence. It uses a hybrid architecture built on the
    Enformer backbone, combining convolutional and self-attention
    layers with a U-net for high-resolution output. Borzoi is trained
    on tiled 524 kb genomic windows and predicts RNA-seq signal in 32
    bp bins across diverse human and mouse biosamples using uniformly
    processed ENCODE and GTEx data. Here we use the pytorch
    implementation of Borzoi from the Gagneur lab.

    Link: https://github.com/calico/borzoi
    Link: https://github.com/johahi/borzoi-pytorch
    """

    max_length = 524_288
    min_length = 196_608
    bin_size = 32  # embedding is in 32 base bins

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return model_version.replace("_", "-")

    def __init__(self, model_version: str, device: torch.device):
        """Initialize Borzoi.

        Args:
            model_version: Version of model used. Valid versions: {
                "borzoi-replicate-0",
                "borzoi-replicate-1",
                "borzoi-replicate-2",
                "borzoi-replicate-3",
                "flashzoi-replicate-0",
                "flashzoi-replicate-1",
                "flashzoi-replicate-2",
                "flashzoi-replicate-3",
                "borzoi",
                "flashzoi"
            }
            device: PyTorch device to send model to.
        """
        super().__init__(model_version, device)

        try:
            from borzoi_pytorch import Borzoi
            from borzoi_pytorch.config_borzoi import BorzoiConfig
        except ImportError:

            if "borzoi" in model_version:
                raise ImportError("Borzoi missing required dependencies.")

            if "flashzoi" in model_version:
                raise ImportError(
                    "Borzoi missing required dependencies."
                    " Flashzoi additionally requires flash attention 2."
                )

        if "flashzoi" in model_version:
            # flashzoi uses float16 precision
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # load ensemble if base model name is given
        if model_version in ["borzoi", "flashzoi"]:

            self.models = []

            for i in range(4):

                sub_version = f"{model_version}-replicate-{i}"

                cfg = BorzoiConfig.from_pretrained(
                    f'johahi/{sub_version}',
                    cache_dir=get_model_weights_path()
                )

                cfg.return_center_bins_only = False

                model_i = Borzoi.from_pretrained(
                    f'johahi/{sub_version}',
                    cache_dir=get_model_weights_path(),
                    config=cfg
                ).to(device, dtype=self.dtype).eval()

                # avoid cropping in the model
                # as we handle padding ourselves per chunk
                model_i.crop = torch.nn.Identity()

                self.models.append(model_i)
        else:

            cfg = BorzoiConfig.from_pretrained(
                f'johahi/{model_version}',
                cache_dir=get_model_weights_path()
            )

            cfg.return_center_bins_only = False

            self.model = Borzoi.from_pretrained(
                f'johahi/{model_version}',
                cache_dir=get_model_weights_path(),
                config=cfg
            ).to(device, dtype=self.dtype).eval()

            self.model.crop = torch.nn.Identity()

    def embed_sequence(
        self,
        sequence: str,
        agg_fn: Callable = torch.mean,
        subset_start: int | None = None,
        subset_end: int | None = None
    ) -> torch.Tensor:
        """Embed sequence using Borzoi, excluding padded regions."""

        def center_padding(seq, length):
            """Center pad sequence to a given length."""
            padding_left = (length - len(seq)) // 2
            padding_right = length - len(seq) - padding_left

            return 'N' * padding_left + seq + 'N' * padding_right, padding_left

        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        with torch.inference_mode():
            for i, chunk in enumerate(chunks):

                if len(chunk) < self.max_length:

                    if len(chunk) < self.min_length:
                        padded_chunk, padding_left = center_padding(chunk, self.min_length)
                    else:
                        # center pad to max length
                        # if chunk is smaller than max length, but larger than min length
                        padded_chunk, padding_left = center_padding(chunk, self.max_length)
                else:
                    padded_chunk, padding_left = chunk, 0

                # first OHE sequence chunk
                batch = torch.tensor(
                    str_to_ohe(padded_chunk),
                    dtype=self.dtype
                ).unsqueeze(0).permute(0, 2, 1).to(self.device)

                # if using ensemble of models, average their embeddings
                if hasattr(self, "models"):
                    replicate_embeds = [model.get_embs_after_crop(batch) for model in self.models]
                    embedded_chunk = torch.stack(replicate_embeds).mean(dim=0)
                else:
                    embedded_chunk = self.model.get_embs_after_crop(batch)

                # extract the portion of the embedding corresponding to original unpadded seq
                start_bin = padding_left // self.bin_size
                end_bin = (padding_left + len(chunk) + (self.bin_size - 1)) // self.bin_size

                embedding = self.subset_sequence_emb(
                    embedded_chunk, start_bin, end_bin
                )

                embedding_chunks.append(embedding)

        embedding = torch.cat(embedding_chunks, dim=2)

        # convert start and end indices to bin indices
        if subset_start is not None:
            subset_start = subset_start // self.bin_size
        if subset_end is not None:
            subset_end = (subset_end + (self.bin_size - 1)) // self.bin_size

        embedding = self.subset_sequence_emb(
            embedding, subset_start, subset_end
        )

        aggregate_embedding = agg_fn(embedding, dim=2)
        return aggregate_embedding

    def embed_sequence_sixtrack(self, sequence, cds, splice):
        """Not supported."""
        raise NotImplementedError("Six track not possible with Borzoi.")