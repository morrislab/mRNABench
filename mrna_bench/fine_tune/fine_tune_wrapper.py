"""Composition-based fine-tuning wrapper for EmbeddingModel."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from mrna_bench.models.embedding_model import EmbeddingModel


class FineTuneWrapper(nn.Module):
    """Wraps any EmbeddingModel for fine-tuning with a task head.

    Holds a backbone EmbeddingModel and a task head nn.Module. Supports
    two mutually exclusive training regimes:

    - apply_lora(): LoRA adapters are applied to the backbone's internal
      model, and the existing embed() path flows through them.
    - freeze_backbone(): the backbone is left untouched and only the
      pooler and head train, i.e. probing with a learned readout.

    The task head can be any nn.Module that also exposes:
        - task_type: str
        - get_loss_fn() -> nn.Module
    """

    _DEFAULT_TRANSFORMER_TARGETS = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "query", "value", "key",
        "Wqkv", "out_proj",
    ]

    def __init__(
        self,
        model: EmbeddingModel,
        task_head: nn.Module,
        pooler: nn.Module | None = None,
    ):
        """Initialize FineTuneWrapper.

        Args:
            model: EmbeddingModel instance to use as backbone.
            task_head: Task head module. Must have task_type and
                get_loss_fn() attributes.
            pooler: Module pooling token embeddings over the sequence
                axis, called as pooler(x, dim=k). None uses torch.mean.
        """
        super().__init__()
        self.backbone = model
        self.task_head = task_head.to(model.device)
        self.pooler = None if pooler is None else pooler.to(model.device)
        self.device = model.device

        self.backbone_frozen = False
        self.lora_applied = False

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

        Raises:
            RuntimeError: If the backbone was already frozen.
        """
        if self.backbone_frozen:
            raise RuntimeError(
                "Backbone is frozen; LoRA adapters would receive no "
                "gradient. Use either freeze_backbone() or apply_lora()."
            )

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

        self.backbone.model = get_peft_model(self.backbone.model, config)

        trainable = sum(
            p.requires_grad for p in self.backbone.model.parameters()
        )
        if trainable == 0:
            raise ValueError(
                f"LoRA applied but no trainable parameters found. "
                f"target_modules={target_modules} may not match any "
                f"layers in {type(self.backbone).__name__}."
            )

        self.lora_applied = True
        self._disable_fast_attention()

    def freeze_backbone(self):
        """Freeze all backbone parameters, training only pooler and head.

        The backbone stays in eval mode and its outputs carry no
        grad_fn, so no backbone activations are stored for backward.
        Gradients still reach the pooler and head, which are applied to
        those outputs as leaf modules.

        Raises:
            RuntimeError: If LoRA adapters were already applied.
        """
        if self.lora_applied:
            raise RuntimeError(
                "LoRA adapters are applied; freezing would leave no "
                "trainable backbone parameters. Use either "
                "freeze_backbone() or apply_lora()."
            )

        for param in self.backbone.model.parameters():
            param.requires_grad = False

        self.backbone_frozen = True

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
            base = self.backbone.model.base_model.model
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
        agg_fn: Callable | None = None,
    ) -> torch.Tensor:
        """Forward pass through backbone and task head.

        Args:
            sequences: List of input sequences.
            cds: CDS tracks for models that use them.
            splice: Splice tracks for models that use them.
            agg_fn: Aggregation function for pooling. None uses the
                pooler given at construction, falling back to torch.mean.

        Returns:
            Task head output tensor.
        """
        if agg_fn is None:
            agg_fn = torch.mean if self.pooler is None else self.pooler

        embeddings = self.backbone.embed(sequences, cds, splice, agg_fn)
        return self.task_head(embeddings)

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get all trainable parameters for optimizer.

        Returns:
            List of trainable parameters (backbone LoRA + head + pooler).
        """
        params = [
            p for p in self.backbone.model.parameters() if p.requires_grad
        ]
        params.extend(self.task_head.parameters())
        if self.pooler is not None:
            params.extend(self.pooler.parameters())
        return params

    def get_parameter_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dictionary with parameter count breakdown.
        """
        total = sum(p.numel() for p in self.backbone.model.parameters())
        backbone_trainable = sum(
            p.numel()
            for p in self.backbone.model.parameters()
            if p.requires_grad
        )
        head_params = sum(
            p.numel() for p in self.task_head.parameters()
        )
        pooler_params = 0
        if self.pooler is not None:
            pooler_params = sum(p.numel() for p in self.pooler.parameters())

        return {
            "backbone_total": total,
            "backbone_trainable": backbone_trainable,
            "head": head_params,
            "pooler": pooler_params,
            "total_trainable": (
                backbone_trainable + head_params + pooler_params
            ),
        }

    def set_train_mode(self):
        """Set head and pooler to training mode with gradients enabled.

        A frozen backbone stays in eval mode so its dropout and
        normalization statistics keep their inference behaviour.
        """
        if self.backbone_frozen:
            self.backbone.model.eval()
            torch.set_grad_enabled(True)
        else:
            self.backbone.set_train_mode()

        self.task_head.train()
        if self.pooler is not None:
            self.pooler.train()

    def set_inference_mode(self):
        """Set backbone, head and pooler to eval mode without gradients."""
        self.backbone.set_inference_mode()
        self.task_head.eval()
        if self.pooler is not None:
            self.pooler.eval()
