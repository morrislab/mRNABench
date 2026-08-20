"""Composition-based fine-tuning wrapper for EmbeddingModel."""

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from mrna_bench.models.embedding_model import EmbeddingModel, mean_pool


class FineTuneWrapper(nn.Module):
    """Wraps any EmbeddingModel for LoRA fine-tuning with a task head.

    Holds a backbone EmbeddingModel and a task head nn.Module.
    LoRA adapters are applied to the backbone's internal model,
    and the existing embed() path naturally flows through them.

    The task head can be any nn.Module that also exposes:
        - task_type: str
        - get_loss_fn() -> nn.Module
    """

    _DEFAULT_TRANSFORMER_TARGETS = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "query", "value", "key",
        "Wqkv", "out_proj",
    ]

    def __init__(self, model: EmbeddingModel, task_head: nn.Module):
        """Initialize FineTuneWrapper.

        Args:
            model: EmbeddingModel instance to use as backbone.
            task_head: Task head module. Must have task_type and
                get_loss_fn() attributes.
        """
        super().__init__()
        self.backbone = model
        self.task_head = task_head.to(model.device)
        self.device = model.device

    def apply_lora(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
    ):
        """Apply LoRA adapters to the backbone model.

        Target modules are resolved in order:
        1. Explicit target_modules argument (if provided)
        2. backbone.lora_target_modules class variable — models with
           non-transformer architectures (e.g. Mamba-based Orthrus)
           declare their own target modules this way
        3. Default transformer projection layer names

        Args:
            rank: Rank of LoRA decomposition.
            alpha: LoRA scaling factor.
            dropout: Dropout probability for LoRA layers.
            target_modules: Module names to apply LoRA to.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "Install peft to use LoRA: pip install peft"
            )

        if target_modules is None:
            target_modules = getattr(
                self.backbone, "lora_target_modules", None
            )
        if target_modules is None:
            target_modules = self._DEFAULT_TRANSFORMER_TARGETS

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            use_rslora=True,
        )

        if isinstance(self.backbone.model, nn.Identity):
            raise NotImplementedError(
                f"{type(self.backbone).__name__} does not support LoRA "
                f"fine-tuning (backbone is nn.Identity)."
            )

        from transformers import PreTrainedModel
        peft_target = self.backbone.get_peft_target()
        peft_model = get_peft_model(
            cast(PreTrainedModel, peft_target), config
        )
        self.backbone.set_peft_target(peft_model)

        trainable = sum(
            p.requires_grad
            for p in self.backbone.get_peft_target().parameters()
        )
        if trainable == 0:
            raise ValueError(
                f"LoRA applied but no trainable parameters found. "
                f"target_modules={target_modules} may not match any "
                f"layers in {type(self.backbone).__name__}."
            )

        self._disable_fast_attention()

    def _disable_fast_attention(self):
        """Disable fast attention paths that bypass LoRA adapters.

        Some model libraries have an optimized attention code path that
        calls F.multi_head_attention_forward() directly with the raw
        weight matrices. This bypasses the layer's forward() method,
        which means LoRA adapters (which wrap forward()) are silently
        skipped.

        Setting enable_torch_version = False forces attention through
        the standard forward() path where LoRA is active. Only affects
        models whose attention layers have this attribute.
        """
        try:
            base = self.backbone.get_peft_target().base_model.model
            if hasattr(base, "layers"):
                for layer in base.layers:
                    if hasattr(layer, "self_attn"):
                        attn = layer.self_attn
                        if hasattr(attn, "enable_torch_version"):
                            attn.enable_torch_version = False
        except AttributeError:
            pass

    def forward(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = mean_pool,
    ) -> torch.Tensor:
        """Forward pass through backbone and task head.

        Args:
            sequences: List of input sequences.
            cds: CDS tracks for models that use them.
            splice: Splice tracks for models that use them.
            agg_fn: Aggregation function for pooling.

        Returns:
            Task head output tensor.
        """
        embeddings = self.backbone.embed(sequences, cds, splice, agg_fn)

        # agg_fns which don't result in a 1-D pooled vector are not supported
        if embeddings[0].dim() != 1:
            raise ValueError(
                f"agg_fn must produce a 1-D pooled vector of shape (H,). "
                f"Got shape {embeddings[0].shape}."
            )

        stacked = torch.stack(embeddings, dim=0).to(self.device)
        return self.task_head(stacked)

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get all trainable parameters for optimizer.

        Returns:
            List of trainable parameters (backbone LoRA + head).
        """
        params = [
            p for p in self.backbone.get_peft_target().parameters()
            if p.requires_grad
        ]
        params.extend(self.task_head.parameters())
        return params

    def get_parameter_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dictionary with parameter count breakdown.
        """
        peft_target = self.backbone.get_peft_target()
        total = sum(p.numel() for p in peft_target.parameters())
        backbone_trainable = sum(
            p.numel()
            for p in peft_target.parameters()
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
        """Set backbone and head to training mode with gradients."""
        self.backbone.set_train_mode()
        self.task_head.train()

    def set_inference_mode(self):
        """Set backbone and head to eval mode without gradients."""
        self.backbone.set_inference_mode()
        self.task_head.eval()
