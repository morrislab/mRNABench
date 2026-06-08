from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mrna_bench import get_model_weights_path
from mrna_bench.models import EmbeddingModel


class RNAMSM(EmbeddingModel):
    """Inference wrapper for RNA-MSM.

    RNA-MSM is a transformer-based RNA foundation model pretrained using custom
    structure-based MSAs between ~4000 RNA families with ~3000 MSAs each.

    Link: https://github.com/yikunpku/RNA-MSM
    """

    default_version = "RNA-MSM"
    valid_versions = ["RNA-MSM"]
    default_attn_implementation = "eager"
    valid_attn_implementations = [
        "eager",
    ]
    hookable_layer_patterns = [r"layers\.\d+"]

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        short_name_map = {
            "RNA-MSM": "rnamsm",
        }
        return short_name_map[model_version]

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
        """Initialize RNA-MSM.

        Args:
            model_version: Must be "RNA-MSM".
            device: PyTorch device to send model to.
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Install base_models optional dependency to use RNA-MSM."
            )

        hub_id = "Taykhoom/{}".format(model_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
        )

        self.model = AutoModel.from_pretrained(
            hub_id,
            trust_remote_code=True,
            cache_dir=get_model_weights_path(),
            attn_implementation=self.attn_implementation,
        ).to(device)
        self.max_length = self.tokenizer.model_max_length

    def _forward_chunks(
        self,
        chunks: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask) tensors.
        """
        toks = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        hidden_states = self.model(**toks).last_hidden_state
        # RNA-MSM returns (B, num_msa, T, D); single sequences have num_msa=1.
        hidden_states = hidden_states[:, 0]

        # input_ids / attention_mask are also (B, num_msa, T); squeeze MSA dim.
        input_ids = toks["input_ids"]
        if input_ids.dim() == 3:
            input_ids = input_ids[:, 0]
        attention_mask = toks["attention_mask"]
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0]

        # RNA-MSM's tokenizer prepends CLS but appends no EOS. Exclude only
        # genuine special tokens (robust if the tokenizer later adds EOS).
        pooling_mask = attention_mask.clone()
        special_ids = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device
        )
        is_special = torch.isin(input_ids, special_ids)
        pooling_mask[is_special] = 0

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = partial(torch.mean, dim=0)
    ) -> list[torch.Tensor]:
        """Embed sequences using RNA-MSM.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate embedding across length dim.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (768,)
        """
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        return self._embed_with_chunking(
            sequences=sequences,
            max_chunk_length=self.max_length - 1,
            embed_fn=self._forward_chunks,
            agg_fn=agg_fn,
        )

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
        """Extract per-layer representations from RNA-MSM.

        Args:
            sequences: RNA sequences (T or U bases; T→U applied internally).
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        # RNA-MSM is an MSA transformer whose hidden_states have shape
        # (batch, num_msa, T, D) instead of the standard (batch, T, D).
        # We process single sequences and squeeze both extra dims.
        _, _ = cds, splice
        sequences = [s.replace("T", "U") for s in sequences]

        resolved = self._resolve_layer_paths(layers)
        hookable = self.hookable_layers
        layer_to_idx = {p: i for i, p in enumerate(hookable)}

        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }
        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: ([] if return_attentions else None) for p in resolved
        }

        for seq in sequences:
            chunks = self.chunk_sequence(seq, self.max_length - 1)
            seq_hidden: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved
            }
            seq_scores: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved
            }

            for chunk in chunks:
                toks = self.tokenizer(
                    [chunk], return_tensors="pt", padding=False
                ).to(self.device)

                outputs = self.model(
                    **toks,
                    output_hidden_states=True,
                    output_attentions=return_attentions,
                )
                # tuple of (B, num_msa, T, D)
                hf_hidden = outputs.hidden_states

                for path in resolved:
                    idx = layer_to_idx[path]
                    hs_idx = idx + 1  # HF offset: index 0 is embedding
                    h = hf_hidden[hs_idx][0, 0]  # squeeze B, num_msa → (T, D)
                    seq_hidden[path].append(h.cpu() if offload_to_cpu else h)

                    if return_attentions and outputs.attentions is not None:
                        # attentions[idx]: (H, num_msa, T, T) — attends over
                        # sequence positions; squeeze the (degenerate) MSA dim
                        # for single-sequence inputs to get (H, T, T).
                        a = outputs.attentions[idx][:, 0]  # (H, T, T)
                        seq_scores[path].append(
                            a.cpu() if offload_to_cpu else a)

            for path in resolved:
                hidden_out[path].append(seq_hidden[path])
                s = score_out[path]
                if s is not None:
                    s.append(seq_scores[path])

        return hidden_out, score_out
