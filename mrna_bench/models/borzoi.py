from collections.abc import Callable
from functools import partial
import warnings
import math

import numpy as np
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

    default_version = "flashzoi"
    valid_versions = [
        "borzoi-replicate-0",
        "borzoi-replicate-1",
        "borzoi-replicate-2",
        "borzoi-replicate-3",
        "flashzoi-replicate-0",
        "flashzoi-replicate-1",
        "flashzoi-replicate-2",
        "flashzoi-replicate-3",
        "borzoi",
        "flashzoi",
    ]
    default_attn_implementation = "flash_attention_2"
    valid_attn_implementations = [
        "eager",
        "flash_attention_2"
    ]

    min_length = 196_608
    bin_size = 32  # embedding is in 32 base bins
    # res_tower CNN blocks -> unet skip block -> transformer blocks ->
    # upsampling/separable CNN blocks. (conv_dna is intentionally excluded.)
    hookable_layer_patterns = [
        # r"conv_dna",
        r"res_tower\.\d+",
        r"unet\d+",
        r"transformer\.\d+",
        r"upsampling_unet\d+\.\d+",
        r"separable\d+",
    ]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
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
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )
        self.max_length = 524_288

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
            self.dtype = torch.float16

            if self.attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Flashzoi model selected with eager attention. "
                    "Flashzoi can only use FA2 so this option will be ignored."
                    "To use this attention backend specify a borzoi replicate"
                    " model version instead."
                )
        else:
            self.dtype = torch.float32

            if self.attn_implementation != "eager":
                warnings.warn(
                    f"Borzoi model version selected with attn_implementation="
                    f"{self.attn_implementation}. Borzoi uses eager attention"
                    " so this option will be ignored. To use flash attention 2"
                    " specify a flashzoi replicate model version instead."
                )

        # load ensemble if base model name is given
        self.models = []

        if model_version in ["borzoi", "flashzoi"]:
            replicate_template = "{}-replicate-{{}}".format(model_version)
            versions_to_load = [replicate_template.format(i) for i in range(4)]
        else:
            versions_to_load = [model_version]

        for version in versions_to_load:
            cfg = BorzoiConfig.from_pretrained(
                "johahi/{}".format(version),
                cache_dir=get_model_weights_path()
            )

            cfg.return_center_bins_only = False

            # initialize empty model that will be filled
            # deals with transformers changes (past v4.51)
            # Sol: https://github.com/huggingface/transformers/issues/28972
            model_i = Borzoi(cfg)

            pretrained_model_i = Borzoi.from_pretrained(
                f"johahi/{version}",
                cache_dir=get_model_weights_path(),
            )

            # assign weights from pretrained model
            model_i.load_state_dict(
                pretrained_model_i.state_dict(),
                strict=True
            )

            model_i = model_i.to(device=device, dtype=self.dtype)

            # Avoid cropping as we handle padding ourselves per chunk
            model_i.crop = torch.nn.Identity()

            self.models.append(model_i)

    @property
    def hookable_layers(self) -> list[str]:
        """Ordered list of hookable layers for the first Borzoi replicate.

        Returns layers in forward-pass order as called by get_embs_after_crop:
        - conv_dna (initial DNA embedding) (ignored for embeddding extraction)
        - res_tower.* (5 residual CNN downsampling blocks at even indices)
        - unet1 (skip-connection ConvBlock applied before max-pool)
        - transformer.* (8 transformer blocks)
        - upsampling_unet*.0 and separable* (2 CNN upsampling blocks each)

        CNN layers output (B, C, T) channel-first; extract() transposes
        these to (T, C) automatically so all hidden-state tensors have a
        consistent (T, D) shape.

        Discovered on self.models[0] (rather than the base-class self.model)
        since Borzoi loads its replicates into self.models.
        """
        from mrna_bench.models.embedding_model import discover_layers
        return discover_layers(self.models[0], self.hookable_layer_patterns)

    def get_peft_target(self) -> torch.nn.Module:
        """Return the first Borzoi replicate for LoRA injection.

        Ensemble versions ("borzoi", "flashzoi") load 4 replicates into
        self.models; single-replicate versions load exactly one. Fine-tuning
        always targets self.models[0] — the remaining replicates are unchanged
        and unused during the LoRA training forward pass.
        """
        return self.models[0]

    def set_peft_target(self, peft_model: torch.nn.Module) -> None:
        """Write the PeftModel back into the first replicate slot.

        Args:
            peft_model: PeftModel returned by get_peft_model().
        """
        self.models[0] = peft_model

    def set_inference_mode(self) -> None:
        """Set all Borzoi replicate models to inference mode."""
        for m in self.models:
            m.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self) -> None:
        """Set all Borzoi replicate models to training mode."""
        for m in self.models:
            m.train()
        torch.set_grad_enabled(True)

    def embed_sequence(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> torch.Tensor:
        """Embed sequence using Borzoi, excluding padded regions.

        Args:
            sequence: Sequence to embed.
            agg_fn: Aggregation function to apply across sequence bins.

        Returns:
            Tensor representing embedded sequence.
        """
        _, _ = cds, splice

        def center_padding(seq: str, length: int) -> tuple[str, int]:
            """Center pad sequence to a given length."""
            padding_left = (length - len(seq)) // 2
            padding_right = length - len(seq) - padding_left

            return "N" * padding_left + seq + "N" * padding_right, padding_left

        chunks = self.chunk_sequence(sequence, self.max_length)

        embedding_chunks = []

        for chunk in chunks:
            if len(chunk) < self.min_length:
                padded_chunk, padding_left = center_padding(
                    chunk,
                    self.min_length
                )
            elif len(chunk) < self.max_length:
                padded_chunk, padding_left = center_padding(
                    chunk,
                    self.max_length
                )
            else:
                padded_chunk, padding_left = chunk, 0

            # first OHE sequence chunk
            batch = torch.tensor(
                str_to_ohe(padded_chunk),
                dtype=self.dtype
            ).unsqueeze(0).permute(0, 2, 1).to(self.device)

            # average embeddings across model replicates
            replicate_embeds = [
                m.get_embs_after_crop(batch) for m in self.models
            ]
            embedded_chunk = torch.stack(replicate_embeds).mean(dim=0)

            # extract embedding portion corresponding to original unpadded seq
            start_bin = padding_left // self.bin_size
            end_bin = math.ceil((padding_left + len(chunk)) / self.bin_size)

            embedding = embedded_chunk[:, :, start_bin:end_bin]

            embedding_chunks.append(embedding.permute(0, 2, 1))

        embedding = torch.cat(embedding_chunks, dim=1).squeeze(0)

        aggregate_embedding = agg_fn(embedding).unsqueeze(0)
        return aggregate_embedding

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0),
    ) -> list[torch.Tensor]:
        """Embed sequences using Borzoi.

        Processes sequences one at a time due to memory constraints
        at 500k+ bp sequence lengths.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (1536,)
        """
        _, _ = cds, splice

        all_embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence, agg_fn=agg_fn)
            all_embeddings.append(embedding.squeeze(0))

        return all_embeddings

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
        """Extract per-layer representations from Borzoi.

        Uses forward hooks on the first replicate model. CNN layers
        (res_tower.*, unet*, upsampling_unet*.*, separable*) output (C, T)
        channel-first; extract() transposes these to (T, C). Transformer
        layers output (T, D) directly.

        Attention weights are available for transformer layers when
        return_attentions=True and the model is a borzoi-replicate-* variant
        (eager attention). flashzoi variants use a fused FlashAttention2
        kernel that discards the attention matrix; a warning is raised and
        scores remain None.

        Args:
            sequences: DNA sequences.
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: If True and model is borzoi-replicate-*,
                transformer layers return (H, T, T) attention tensors.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); scores[path] is None for CNN/skip layers
            and list[list[Tensor(H,T,T)]] for eager-attention
            transformer layers.
        """
        _, _ = cds, splice

        # Use the first replicate model for layer extraction
        primary_model = self.models[0]

        resolved = self._resolve_layer_paths(layers)
        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }

        # Flashzoi uses FlashAttention2 — the fused kernel discards
        # the attention matrix
        if "flashzoi" in self.model_version and return_attentions:
            warnings.warn(
                "Attention extraction is not supported for flashzoi models "
                "(FlashAttention2 discards the attention matrix). "
                "Use a borzoi-replicate-* version to extract "
                "attention weights."
            )
            return_attentions = False

        # Eligible paths: top-level transformer blocks with an
        # attn_dropout submodule
        attn_paths = set()
        if return_attentions:
            for p in resolved:
                if p.startswith("transformer.") and p.count(".") == 1:
                    try:
                        primary_model.get_submodule(p + ".0.fn.1.attn_dropout")
                        attn_paths.add(p)
                    except AttributeError:
                        pass  # unexpected structure — skip silently

        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: ([] if p in attn_paths else None) for p in resolved
        }

        def center_padding(seq: str, length: int) -> tuple[str, int]:
            padding_left = (length - len(seq)) // 2
            padding_right = length - len(seq) - padding_left
            return "N" * padding_left + seq + "N" * padding_right, padding_left

        # Register hooks on the primary model's layers
        handles = []
        activations_store: dict[str, list[torch.Tensor]] = {
            p: [] for p in resolved
        }
        for path in resolved:
            module = primary_model.get_submodule(path)

            def make_hook(name: str):
                def hook(_, __, output):
                    out = output[0] if isinstance(output, tuple) else output
                    activations_store[name].append(out.detach())
                return hook

            handles.append(module.register_forward_hook(make_hook(path)))

        # Register input hooks on attn_dropout to capture
        # (B, H, T, T) attention matrices
        score_store: dict[str, list[torch.Tensor]] = {
            p: [] for p in attn_paths
        }
        for path in attn_paths:
            attn_drop_mod = primary_model.get_submodule(
                path + ".0.fn.1.attn_dropout"
            )

            def make_attn_hook(name: str):
                def attn_hook(
                    _: torch.nn.Module,
                    inp: tuple,
                    __: torch.Tensor,
                ) -> None:
                    score_store[name].append(inp[0].detach())
                return attn_hook

            handles.append(
                attn_drop_mod.register_forward_hook(make_attn_hook(path))
            )

        try:
            for seq in sequences:
                chunks = self.chunk_sequence(seq, self.max_length)
                seq_hidden: dict[str, list[torch.Tensor]] = {
                    p: [] for p in resolved
                }
                seq_score: dict[str, list[torch.Tensor]] = {
                    p: [] for p in attn_paths
                }

                for chunk in chunks:
                    if len(chunk) < self.min_length:
                        padded_chunk, _ = center_padding(
                            chunk, self.min_length
                        )
                    elif len(chunk) < self.max_length:
                        padded_chunk, _ = center_padding(
                            chunk, self.max_length
                        )
                    else:
                        padded_chunk = chunk

                    batch = torch.tensor(
                        str_to_ohe(padded_chunk),
                        dtype=self.dtype,
                    ).unsqueeze(0).permute(0, 2, 1).to(self.device)

                    # Clear previous activations
                    for p in resolved:
                        activations_store[p].clear()
                    for p in attn_paths:
                        score_store[p].clear()

                    primary_model.get_embs_after_crop(batch)

                    for path in resolved:
                        if activations_store[path]:
                            h = activations_store[path][0]
                            if h.dim() == 3:
                                h = h[0]  # squeeze batch dim
                                # CNN layers are (C, T) after squeeze;
                                # transformer layers are already (T, D).
                                if not path.startswith("transformer"):
                                    h = h.T  # (C, T) → (T, C)
                            seq_hidden[path].append(
                                h.cpu() if offload_to_cpu else h
                            )

                    for path in attn_paths:
                        if score_store[path]:
                            w = score_store[path][0][0]  # (1,H,T,T) → (H,T,T)
                            seq_score[path].append(
                                w.cpu() if offload_to_cpu else w)

                for path in resolved:
                    hidden_out[path].append(seq_hidden[path])
                for path in attn_paths:
                    score_list = score_out[path]
                    assert isinstance(score_list, list)
                    score_list.append(seq_score[path])

        finally:
            for handle in handles:
                handle.remove()

        return hidden_out, score_out
