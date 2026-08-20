from collections.abc import Callable
from typing import Optional
import warnings

import torch
import numpy as np

from mrna_bench import get_model_weights_path
from mrna_bench.models.embedding_model import (
    EmbeddingModel,
    ModelBehavior,
    TrackOutput,
)


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
    of the model were further trained to predict 16K+ genomic tracks.

    Link: https://github.com/instadeepai/nucleotide-transformer
    """

    default_version = "v3_650M_post"
    valid_versions = [
        "v3_8M_pre",
        "v3_100M_pre",
        "v3_650M_pre",
        "v3_100M_post",
        "v3_650M_post",
    ]
    default_attn_implementation = "eager"
    valid_attn_implementations = [
        "eager",
    ]
    # NTV3's output_hidden_states returns one tensor per U-Net block in
    # named_modules() order, which matches the forward order:
    # conv_tower_blocks (0..N-1) -> transformer_blocks (0..M-1) ->
    # deconv_tower_blocks (0..N-1). No initial embedding entry is included.
    hookable_layer_patterns = [
        r"core\.conv_tower_blocks\.\d+",
        r"core\.transformer_blocks\.\d+",
        r"core\.deconv_tower_blocks\.\d+",
    ]
    supported_behaviors = frozenset({ModelBehavior.EMBEDDING})

    @classmethod
    def behaviors_for_version(
        cls,
        model_version: str,
    ) -> frozenset[ModelBehavior]:
        """Return pretraining- or post-training-specific behaviors.

        Args:
            model_version: Version whose supported behaviors to retrieve.

        Returns:
            Embedding plus pseudo-likelihood for pre-trained checkpoints, or
            embedding plus tracks for post-trained checkpoints.
        """
        behaviors = set(super().behaviors_for_version(model_version))
        behaviors.add(
            ModelBehavior.TRACKS
            if "post" in model_version
            else ModelBehavior.PSEUDO_LIKELIHOOD
        )
        return frozenset(behaviors)

    @staticmethod
    def get_model_short_name(model_version: str) -> str:
        """Get shortened name of model version."""
        return "nt-" + model_version.replace("_", "-")

    def __init__(
        self,
        model_version: str,
        device: torch.device,
        attn_implementation: str | None,
    ):
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
            attn_implementation: Attention backend.
        """
        super().__init__(
            model_version,
            device,
            attn_implementation
        )
        self.max_length = 1_000_064

        self.post_trained = "post" in model_version

        try:
            if self.post_trained:
                from transformers import AutoModel
            else:
                from transformers import AutoModelForMaskedLM

            from transformers import AutoTokenizer, AutoConfig

        except ImportError:
            raise ImportError((
                "Install base_models optional dependency to use "
                "NucleotideTransformerV3."
            ))

        self.config = AutoConfig.from_pretrained(
            "InstaDeepAI/NT{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/NT{}".format(model_version),
            trust_remote_code=True,
            cache_dir=get_model_weights_path()
        )

        if self.post_trained:
            self.model = AutoModel.from_pretrained(
                "InstaDeepAI/NT{}".format(model_version),
                config=self.config,
                trust_remote_code=True,
                cache_dir=get_model_weights_path(),
                attn_implementation=self.attn_implementation,
            ).to(self.device)

            s_ids = self.config.species_to_token_id.keys()

            # a species token is required for post-trained models
            self.valid_species = [
                species for species in s_ids
                if not species.startswith("<")
            ]
            self.species_id: Optional[torch.Tensor] = None
        else:
            language_model = AutoModelForMaskedLM.from_pretrained(
                "InstaDeepAI/NT{}".format(model_version),
                config=self.config,
                trust_remote_code=True,
                cache_dir=get_model_weights_path(),
                attn_implementation=self.attn_implementation,
            ).to(self.device)
            self._set_logits_model(language_model)
            self.sequence_score_chunk_length = self.max_length

    def _tokenize_for_logits(
        self,
        sequence: str,
        cds: np.ndarray | None = None,
        splice: np.ndarray | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Pad scoring inputs to NTV3's required 128-token multiple.

        Args:
            sequence: Nucleotide sequence to tokenize.
            cds: Unused.
            splice: Unused.
            add_special_tokens: Unused; NTV3 scoring omits special tokens.

        Returns:
            Tokenizer inputs padded to a multiple of 128 tokens.
        """
        _ = cds, splice, add_special_tokens
        tokenized = self.tokenizer(
            sequence,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
        )
        tokenized["attention_mask"] = tokenized["input_ids"].ne(
            self.tokenizer.pad_token_id
        ).long()
        return tokenized  # type: ignore[no-any-return]

    def set_species(self, species: str):
        """Set species for post-trained NucleotideTransformerV3 model.

        Args:
            species: Species name. Must be one of the valid species names
                supported by the model.
        """
        if not self.post_trained:
            warnings.warn((
                "Setting species is only supported for post-trained "
                "NucleotideTransformerV3 models. Ignoring species input."
            ))
            return

        if species not in self.valid_species:
            warnings.warn((
                f"'{species}' sequences do not map to a dedicated species "
                "token in NucleotideTransformerV3 post-trained models. "
                "Using the 'human' species token instead."
            ))
            species = "human"

        self.species_id = self.model.encode_species(  # type: ignore[operator]
            [species]
        ).to(self.device)

    def _forward_chunks(
        self,
        chunks: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of sequence chunks.

        Args:
            chunks: List of sequence chunks to embed.

        Returns:
            Tuple of (hidden_states, pooling_mask). The pooling_mask excludes
            padding tokens. NTV3 does not use special tokens like CLS/SEP.
        """
        toks = self.tokenizer(
            chunks,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
            return_special_tokens_mask=True
        ).to(self.device)

        if self.post_trained:

            assert self.species_id is not None

            s_ids = self.species_id.expand(
                len(chunks)
            )

            hidden_states = self.model(
                species_ids=s_ids,
                input_ids=toks["input_ids"],
                output_hidden_states=True
            ).hidden_states[-1]
        else:
            hidden_states = self.model(
                input_ids=toks["input_ids"],
                output_hidden_states=True
            ).hidden_states[-1]

        pooling_mask = 1 - toks["special_tokens_mask"]

        return hidden_states, pooling_mask

    def embed(
        self,
        sequences: list[str],
        cds: list[np.ndarray] | None = None,
        splice: list[np.ndarray] | None = None,
        agg_fn: Callable = EmbeddingModel.mean_pool
    ) -> list[torch.Tensor]:
        """Embed sequences using NucleotideTransformerV3.

        Each sequence is processed in its own forward pass to avoid the
        U-Net's convolutional avg_pool layers producing context-dependent
        outputs when sequences of different lengths are padded together.

        Args:
            sequences: List of sequences to embed.
            cds: Unused.
            splice: Unused.
            agg_fn: Function used to aggregate token embeddings.

        Returns:
            Embeddings with item shape depending on agg_fn.
            - default (mean): (256,) for `v3_8M_pre`
            - default (mean): (768,) for `v3_100M_pre/post`
            - default (mean): (1536,) for `v3_650M_pre/post`
        """
        _, _ = cds, splice

        if self.post_trained and self.species_id is None:

            self.set_species("human")

            warnings.warn((
                "Species must be set for post-trained NucleotideTransformerV3 "
                "models. Using default species_id for embedding ('human')."
                " Use the `set_species` method to change the species."
            ))

        return [
            self._embed_with_chunking(
                sequences=[seq],
                max_chunk_length=self.max_length,
                embed_fn=self._forward_chunks,
                agg_fn=agg_fn,
            )[0]
            for seq in sequences
        ]

    def predict_tracks(
        self,
        sequences: list[str],
    ) -> list[TrackOutput]:
        """Return post-trained NTV3 BigWig and BED tracks.

        Args:
            sequences: Nucleotide sequences to predict.

        Returns:
            Post-trained model tracks aligned to each input sequence.
        """
        if not self.post_trained:
            raise ValueError(
                "Track prediction requires a post-trained NTV3 checkpoint."
            )
        if any(len(sequence) > self.max_length for sequence in sequences):
            raise ValueError(
                "predict_tracks accepts one NTV3 window per sequence."
            )
        if self.species_id is None:
            self.set_species("human")
            warnings.warn(
                "Species was not set; using the human track head."
            )

        outputs = []
        for sequence in sequences:
            input_ids = self.tokenizer(
                sequence,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(self.device)
            token_length = input_ids.shape[1]
            padded_length = (
                (token_length + 127) // 128
            ) * 128
            padding_left = (padded_length - token_length) // 2
            padding_right = (
                padded_length - token_length - padding_left
            )
            input_ids = torch.nn.functional.pad(
                input_ids,
                (padding_left, padding_right),
                value=self.tokenizer.pad_token_id,
            )
            assert self.species_id is not None
            predictions = self.model(
                input_ids=input_ids,
                species_ids=self.species_id.expand(1),
                output_track=True,
            )
            bigwig = predictions.bigwig_tracks_logits
            bed = predictions.bed_tracks_logits
            if bigwig is None and bed is None:
                raise RuntimeError("NTV3 returned no track predictions.")
            track_length = (
                bigwig.shape[1] if bigwig is not None else bed.shape[1]
            )
            crop_start = (padded_length - track_length) // 2
            overlap_start = max(crop_start, padding_left)
            overlap_end = min(
                crop_start + track_length,
                padding_left + token_length,
            )
            output_start = overlap_start - crop_start
            output_end = overlap_end - crop_start
            values = {}
            if bigwig is not None:
                values["bigwig"] = bigwig[0, output_start:output_end]
            if bed is not None:
                values["bed"] = bed[0, output_start:output_end]
            outputs.append(TrackOutput(
                values,
                overlap_start - padding_left,
                1,
            ))
        return outputs

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
        """Extract per-layer representations from NucleotideTransformerV3.

        NTV3 uses a CNN+Transformer U-Net architecture. Intermediate
        hidden states from CNN layers will have reduced sequence lengths
        due to downsampling. Each sequence is processed individually to
        avoid context-dependent artifacts from padding interactions.

        NTV3's output_hidden_states is indexed directly (no +1 BERT offset):
        hidden_states[i] corresponds to hookable_layers[i].

        Attention weights are available only for transformer_blocks paths;
        all other paths return None for scores.

        Post-trained models default to 'human' species if set_species()
        has not been called.

        Args:
            sequences: DNA sequences (minimum ~128 nt due to U-Net pooling).
            cds: Unused.
            splice: Unused.
            layers: Layer selection; see EmbeddingModel.extract().
            return_attentions: Whether to extract attention weights.
            offload_to_cpu: Move tensors to CPU after each chunk.

        Returns:
            (hidden_states, scores); see EmbeddingModel.extract().
        """
        _, _ = cds, splice

        if self.post_trained and self.species_id is None:
            self.set_species("human")
            warnings.warn((
                "Species must be set for post-trained NucleotideTransformerV3 "
                "models. Using default species_id for embedding ('human')."
                " Use the `set_species` method to change the species."
            ))

        hookable = self.hookable_layers
        resolved = self._resolve_layer_paths(layers)
        layer_to_idx = {p: i for i, p in enumerate(hookable)}

        # Paths belonging to the transformer tower (attention available there)
        transformer_paths = {p for p in hookable if "transformer_blocks" in p}
        # Among transformer paths, their local attention index
        transformer_hookable = [
            p for p in hookable if "transformer_blocks" in p]

        hidden_out: dict[str, list[list[torch.Tensor]]] = {
            p: [] for p in resolved
        }
        score_out: dict[str, list[list[torch.Tensor]] | None] = {
            p: ([] if (return_attentions and p in transformer_paths) else None)
            for p in resolved
        }

        for seq in sequences:
            chunks = self.chunk_sequence(seq, self.max_length)
            seq_hidden: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved
            }
            seq_scores: dict[str, list[torch.Tensor]] = {
                p: [] for p in resolved if p in transformer_paths
            }

            for chunk in chunks:
                toks = self.tokenizer(
                    [chunk],
                    add_special_tokens=False,
                    padding=True,
                    pad_to_multiple_of=128,
                    return_tensors="pt",
                    return_special_tokens_mask=False,
                )
                if self.post_trained:
                    assert self.species_id is not None
                    toks["species_ids"] = self.species_id.expand(
                        1)  # type: ignore[assignment]
                toks = toks.to(self.device)  # type: ignore[assignment]

                with torch.inference_mode():
                    outputs = self.model(
                        **toks,
                        output_hidden_states=True,
                        output_attentions=(
                            return_attentions and bool(transformer_paths)),
                    )
                hf_hidden = outputs.hidden_states
                # attentions is a tuple of (B, H, T, T) per transformer layer
                hf_attns = outputs.attentions if return_attentions else None

                for path in resolved:
                    hs_idx = layer_to_idx[path]
                    h = hf_hidden[hs_idx][0].detach()
                    seq_hidden[path].append(h.cpu() if offload_to_cpu else h)

                    if return_attentions and path in transformer_paths:
                        attn_local = transformer_hookable.index(path)
                        if hf_attns is not None and attn_local < len(hf_attns):
                            a = hf_attns[attn_local][0].detach()
                            seq_scores[path].append(
                                a.cpu() if offload_to_cpu else a
                            )

            for path in resolved:
                hidden_out[path].append(seq_hidden[path])
                s = score_out[path]
                if s is not None and path in seq_scores:
                    s.append(seq_scores[path])

        return hidden_out, score_out
