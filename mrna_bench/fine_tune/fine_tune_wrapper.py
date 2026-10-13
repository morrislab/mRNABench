"""Composition-based fine-tuning wrapper for EmbeddingModel."""

import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from mrna_bench.models.embedding_model import EmbeddingModel
from mrna_bench.models.pooling import MeanPool


class FineTuneWrapper(nn.Module):
    """Wraps any EmbeddingModel for fine-tuning with a task head.

    Holds a backbone EmbeddingModel and a task head nn.Module. The backbone
    is frozen except for ``trainable_layers``, and LoRA adapters can be
    added with apply_lora(); the existing embed() path flows through both.

    Heads with ``per_position`` set (e.g. AttentionPoolHead) receive the
    list of per-sequence (n, H) embeddings, so ``agg_fn`` should not pool
    (e.g. ``NoPool`` or ``PositionPool``).

    The task head can be any nn.Module that also exposes:
        - task_type: str
        - get_loss_fn() -> nn.Module
    """

    def __init__(
        self,
        model: EmbeddingModel,
        task_head: nn.Module,
        trainable_layers: str | list[int | str] | None = None,
        agg_fn: Callable = MeanPool(),
    ):
        """Initialize FineTuneWrapper.

        Freezes all backbone parameters except ``trainable_layers``. Call
        apply_lora() to add trainable LoRA adapters; with neither, only the
        head is trained.

        Args:
            model: EmbeddingModel instance to use as backbone.
            task_head: Task head module. Must have task_type and
                get_loss_fn() attributes.
            trainable_layers: Backbone layers to fine-tune: "all" for full
                fine-tuning, or indices / paths into the model's
                hookable_layers (e.g. ``[-2, -1]`` for the last two blocks,
                plus "final" for a final norm). None trains no backbone
                layers.
            agg_fn: Aggregation function used to embed sequences.
        """
        super().__init__()
        self.backbone = model
        self.task_head = task_head.to(model.device)
        self.device = model.device
        self.agg_fn = agg_fn

        targets = self.backbone.get_peft_targets()
        for target in targets:
            for param in target.parameters():
                param.requires_grad = False

        if trainable_layers == "all":
            modules = targets
        else:
            modules = [
                self.backbone.model.get_submodule(path)
                for path in self.backbone._resolve_layer_paths(
                    cast(list[int | str], trainable_layers or [])
                )
            ]
        # Trained weights are kept in float32: half-precision weights do not
        # survive optimizer updates. Kept so apply_lora(), which freezes the
        # base model, can restore them.
        self.unfrozen_params = [
            p for module in modules for p in module.float().parameters()
        ]
        for param in self.unfrozen_params:
            param.requires_grad = True

    def apply_lora(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
        layers: list[int | str] | None = None,
    ):
        """Apply LoRA adapters to the backbone model.

        Adapters go on the nn.Linear modules whose path ends with one of
        ``target_modules``, which defaults to the model's
        ``lora_target_modules``. For models with multiple sub-models (e.g.
        Borzoi ensemble), LoRA is applied to each one independently.

        Args:
            rank: Rank of LoRA decomposition.
            alpha: LoRA scaling factor.
            dropout: Dropout probability for LoRA layers.
            target_modules: Module names to apply LoRA to.
            layers: Indices or paths into the model's hookable_layers (or
                "final") to restrict adapters to. None applies them in every
                layer.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "Install peft to use LoRA: pip install peft"
            )

        targets = self.backbone.get_peft_targets()
        name = type(self.backbone).__name__

        if any(isinstance(t, nn.Identity) for t in targets):
            raise NotImplementedError(
                f"{name} does not support LoRA "
                f"fine-tuning (backbone is nn.Identity)."
            )

        target_modules = target_modules or self.backbone.lora_target_modules
        if target_modules is None:
            raise NotImplementedError(
                f"{name} declares no lora_target_modules; pass "
                f"target_modules to apply LoRA."
            )

        prefixes = [""]
        if layers is not None:
            prefixes = [
                path + "." for path in
                self.backbone._resolve_layer_paths(layers)
            ]

        def selected(path: str, module: nn.Module) -> bool:
            in_layers = any(path.startswith(p) for p in prefixes)
            named = any(
                path == m or path.endswith("." + m) for m in target_modules
            )
            return isinstance(module, nn.Linear) and in_layers and named

        # Full module paths, so a name shared with a non-linear module
        # (e.g. an activation at the same position) does not match it.
        lora_targets = sorted({
            path
            for t in targets
            for path, module in t.named_modules()
            if selected(path, module)
        })
        if not lora_targets:
            raise ValueError(
                f"target_modules={target_modules} match no linear layers "
                f"in {name}."
            )

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=lora_targets,
            use_rslora=True,
        )

        if len(targets) > 1:
            warnings.warn(
                "{} has {} sub-models — LoRA will be applied to each "
                "one. Expect ~{}x the GPU memory of a single model.".format(
                    name, len(targets), len(targets),
                )
            )

        from transformers import PreTrainedModel
        peft_models = [
            get_peft_model(cast(PreTrainedModel, t), config)
            for t in targets
        ]
        self.backbone.set_peft_targets(
            cast(list[torch.nn.Module], peft_models)
        )

        for param in self.unfrozen_params:
            param.requires_grad = True

        self._disable_fast_attention()

    def _disable_fast_attention(self):
        """Disable fast paths that bypass LoRA adapters.

        Some model libraries have optimized code paths that use the raw
        weight matrices directly, e.g. F.multi_head_attention_forward() for
        attention or mamba_ssm's fused Mamba kernel. This bypasses the
        layer's forward() method, which means LoRA adapters (which wrap
        forward()) are silently skipped.

        Setting enable_torch_version / use_fast_path to False forces these
        layers through the standard forward() path where LoRA is active.
        Only affects modules that have these attributes.
        """
        for target in self.backbone.get_peft_targets():
            for module in target.modules():
                if hasattr(module, "use_fast_path"):
                    module.use_fast_path = False
            try:
                base = target.base_model.model
                if hasattr(base, "layers"):
                    for layer in base.layers:
                        if hasattr(layer, "self_attn"):
                            attn = layer.self_attn
                            if hasattr(attn, "enable_torch_version"):
                                attn.enable_torch_version = False
            except AttributeError:
                pass

    def _embed_and_stack(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = MeanPool(),
    ) -> torch.Tensor:
        """Embed sequences and stack into a batch tensor.

        Args:
            sequences: List of input sequences.
            cds: CDS tracks for models that use them.
            splice: Splice tracks for models that use them.
            agg_fn: Aggregation function for pooling.

        Returns:
            Stacked embeddings of shape (batch, hidden_dim).
        """
        embeddings = self.backbone.embed(sequences, cds, splice, agg_fn)

        if embeddings[0].dim() != 1:
            raise ValueError(
                f"agg_fn must produce a 1-D pooled vector of shape (H,). "
                f"Got shape {embeddings[0].shape}."
            )

        return torch.stack(embeddings, dim=0).to(self.device)

    def _head_forward(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None,
        splice: list[np.ndarray] | None,
        agg_fn: Callable,
    ) -> torch.Tensor:
        """Embed sequences and apply the task head.

        Heads with ``per_position`` set get one (n, H) tensor per sequence
        and pool or score positions themselves; others get a stacked
        (batch, H) tensor of pooled embeddings.
        """
        if getattr(self.task_head, "per_position", False):
            embeddings = self.backbone.embed(sequences, cds, splice, agg_fn)
            return self.task_head([e.to(self.device) for e in embeddings])
        stacked = self._embed_and_stack(sequences, cds, splice, agg_fn)
        return self.task_head(stacked)

    def forward(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Forward pass through backbone and task head.

        Args:
            sequences: List of input sequences.
            cds: CDS tracks for models that use them.
            splice: Splice tracks for models that use them.
            agg_fn: Aggregation function for pooling. Defaults to the
                wrapper's agg_fn.

        Returns:
            Task head output tensor.
        """
        return self._head_forward(
            sequences, cds, splice, agg_fn or self.agg_fn
        )

    def forward_vep(
        self,
        ref_sequences: list[str],
        alt_sequences: list[str],
        ref_cds: list[np.ndarray] | None = None,
        alt_cds: list[np.ndarray] | None = None,
        ref_splice: list[np.ndarray] | None = None,
        alt_splice: list[np.ndarray] | None = None,
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """VEP forward: embed ref and alt separately, return score difference.

        Args:
            ref_sequences: Reference (wild-type) sequences.
            alt_sequences: Alternate (variant) sequences.
            ref_cds: Reference CDS tracks.
            alt_cds: Alternate CDS tracks.
            ref_splice: Reference splice tracks.
            alt_splice: Alternate splice tracks.
            agg_fn: Aggregation function for pooling. Defaults to the
                wrapper's agg_fn.

        Returns:
            Difference of task head predictions (alt - ref).
        """
        agg_fn = agg_fn or self.agg_fn
        preds_ref = self._head_forward(
            ref_sequences, ref_cds, ref_splice, agg_fn
        )
        preds_alt = self._head_forward(
            alt_sequences, alt_cds, alt_splice, agg_fn
        )
        return preds_alt - preds_ref

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get all trainable parameters for optimizer.

        Returns:
            List of trainable parameters (backbone LoRA + head).
        """
        params = [
            p
            for target in self.backbone.get_peft_targets()
            for p in target.parameters()
            if p.requires_grad
        ]
        params.extend(self.task_head.parameters())
        return params

    def trainable_state_dict(self) -> dict:
        """Copy the trained weights: backbone layers and adapters, and head.

        Returns:
            Dictionary with per-target backbone states and head state dict.
        """
        backbone_states = []
        for target in self.backbone.get_peft_targets():
            trainable = {
                name for name, parameter in target.named_parameters()
                if parameter.requires_grad
            }
            backbone_states.append({
                name: value.detach().clone()
                for name, value in target.state_dict().items()
                if name in trainable
            })
        head_state = {
            name: value.detach().clone()
            for name, value in self.task_head.state_dict().items()
        }
        return {"lora": backbone_states, "head": head_state}

    def load_trainable_state_dict(self, state: dict):
        """Restore weights saved by trainable_state_dict().

        Args:
            state: Dictionary with per-target backbone states and head state.
        """
        targets = self.backbone.get_peft_targets()
        for target, backbone_state in zip(targets, state["lora"]):
            target.load_state_dict(backbone_state, strict=False)

        self.task_head.load_state_dict(state["head"])

    def get_parameter_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dictionary with parameter count breakdown.
        """
        targets = self.backbone.get_peft_targets()
        total = sum(
            p.numel() for t in targets for p in t.parameters()
        )
        backbone_trainable = sum(
            p.numel()
            for t in targets
            for p in t.parameters()
            if p.requires_grad
        )
        head_params = sum(
            p.numel() for p in self.task_head.parameters()
        )

        return {
            "backbone_total": total,
            "backbone_trainable": backbone_trainable,
            "head": head_params,
            "total_trainable": backbone_trainable + head_params,
        }

    def set_train_mode(self):
        """Set backbone and head to training mode with gradients.

        BatchNorm layers stay in eval mode so they keep normalizing with
        their pretrained running statistics, which small, padded fine-tuning
        batches would otherwise overwrite.
        """
        self.backbone.set_train_mode()
        for target in self.backbone.get_peft_targets():
            for module in target.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        self.task_head.train()

    def set_inference_mode(self):
        """Set backbone and head to eval mode without gradients."""
        self.backbone.set_inference_mode()
        self.task_head.eval()
