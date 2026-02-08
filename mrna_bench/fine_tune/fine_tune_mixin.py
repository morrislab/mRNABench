from typing import Callable

import numpy as np

import torch
import torch.nn as nn

from mrna_bench.models.embedding_model import SupportsEmbedding


class FineTuneMixin(SupportsEmbedding):
    """Mixin to add fine-tuning capabilities to EmbeddingModel subclasses.

    This mixin enables LoRA-based fine-tuning of genomic foundation models
    with support for task-specific prediction heads and gradient checkpointing.
    """

    def __init__(self, model_version: str, device: torch.device):
        """Initialize with gradients disabled by default.

        Args:
            model_version: Version of the model to load.
            device: PyTorch device for model.
        """
        super().__init__(model_version, device)  # type: ignore[call-arg]
        self._task_head: nn.Module | None = None
        self._lora_applied: bool = False

    def attach_head(self, head: nn.Module):
        """Attach a task-specific prediction head.

        Args:
            head: Task head module for prediction.
        """
        self._task_head = head.to(self.device)

    def apply_lora(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
    ):
        """Apply LoRA adapters to the backbone model.

        Args:
            rank: Rank of LoRA decomposition.
            alpha: LoRA scaling factor.
            dropout: Dropout probability for LoRA layers.
            target_modules: List of module names to apply LoRA to.
                Uses Mamba modules for Orthrus, Transformer modules otherwise.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "Install peft to use LoRA: pip install peft"
            )

        if target_modules is None:
            if self.__class__.__name__ == "FineTunableOrthrus":
                target_modules = ["in_proj", "out_proj", "x_proj", "dt_proj"]
            else:
                target_modules = [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "query", "value", "key",
                    "Wqkv", "out_proj",
                ]

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            use_rslora=True,
        )

        self.model = get_peft_model(self.model, config)
        self._lora_applied = True
        self._disable_fast_attention()

    def _disable_fast_attention(self):
        """Disable fast attention paths that bypass LoRA in fm library models.

        The fm library (used by RNA-FM, mRNA-FM) has a fast attention path that
        directly accesses weight matrices via F.multi_head_attention_forward,
        bypassing LoRA's forward() method. This disables that path to ensure
        LoRA adapters are actually used.
        """
        try:
            base = self.model.base_model.model
            if hasattr(base, "layers"):
                for layer in base.layers:
                    if hasattr(layer, "self_attn"):
                        if hasattr(layer.self_attn, "enable_torch_version"):
                            layer.self_attn.enable_torch_version = False
        except AttributeError:
            pass

    def _count_trainable(self, module: nn.Module) -> int:
        """Count trainable parameters in a module.

        Args:
            module: PyTorch module to count parameters in.
        """
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def get_parameter_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts.

        Returns:
            Dictionary with parameter count breakdown.
        """
        total = sum(p.numel() for p in self.model.parameters())
        backbone_trainable = self._count_trainable(self.model)

        head_params = 0
        if self._task_head is not None:
            head_params = self._count_trainable(self._task_head)

        return {
            "backbone_total": total,
            "backbone_trainable": backbone_trainable,
            "head": head_params,
            "total_trainable": backbone_trainable + head_params,
        }

    def get_trainable_parameters(self):
        """Get all trainable parameters for optimizer.

        Returns:
            Iterator of trainable parameters (backbone + head).
        """
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param)
        if self._task_head is not None:
            for param in self._task_head.parameters():
                params.append(param)
        return params

    def forward_with_head(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        """Forward pass through backbone and task head.

        Args:
            sequences: List of input sequences.
            cds: CDS tracks for models that use it.
            splice: Splice tracks for models that use it.
            agg_fn: Aggregation function for pooling.

        Returns:
            Task head output tensor.
        """
        if self._task_head is None:
            raise RuntimeError("No task head attached. Use attach_head().")

        embeddings = self.embed(sequences, cds, splice, agg_fn)
        return self._task_head(embeddings)


def make_fine_tunable(model_cls: type) -> type:
    """Create a fine-tunable version of a model class.

    Args:
        model_cls: EmbeddingModel subclass to make fine-tunable.

    Returns:
        New class with FineTuneMixin capabilities.
    """
    class FineTunableModel(FineTuneMixin, model_cls):
        pass

    FineTunableModel.__name__ = "FineTunable{}".format(model_cls.__name__)
    FineTunableModel.__qualname__ = "FineTunable{}".format(model_cls.__name__)

    return FineTunableModel
