# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

mRNABench is a benchmark suite for evaluating genomic foundation models on mRNA prediction tasks. It provides 23 datasets, 34 model wrappers, and pipelines for embedding, linear probing, and LoRA fine-tuning.

## Commands

### Installation
```bash
# Development install with model support
pip install -e .[base_models,dev]

# Also install fine-tuning dependencies (LoRA via PEFT)
pip install -e .[fine_tune]
```

### Testing
```bash
# Run the full suite (model tests download weights and need a GPU).
pytest

# Run a single test file
pytest tests/models/test_rnafm.py

# Run a single test
pytest tests/models/test_rnafm.py::test_embed
```

Notes:
- `tests/models/test_helix_mrna.py` and `tests/models/test_ntv3.py` load **gated**
  HuggingFace repos (`helical-ai/Helix-mRNA`, `InstaDeepAI/NTv3_*`). They require
  `hf auth login` with a token that has been granted access; otherwise they 401.

### Linting & Type Checking
```bash
# From precommit.sh - run before committing (runs flake8, mypy, then pytest)
bash precommit.sh

# Individually:
flake8 mrna_bench --ignore=D100,D104,E203
mypy mrna_bench --ignore-missing-imports
```

### Embedding a Dataset
```bash
python scripts/embedding/embed_dataset.py \
    --model_name rnafm \
    --dataset_name rnahl-human \
    --d_chunk_ind 0 --d_num_chunks 1
```

### Running a Linear Probe
```bash
python scripts/linear_probe/by_modelname.py --model_name rnafm --dataset_name rnahl-human
```

### Running Fine-Tuning
```bash
python scripts/finetune/run_finetune.py --model_name orthrus --dataset_name rnahl-human
python scripts/finetune/test_pipeline.py  # Quick sanity check
python scripts/finetune/summarize_ft_results.py  # Aggregate result JSONs into CSV
```

### Regression Test
```bash
python scripts/regression_test.py  # End-to-end pipeline smoke test
```

## Architecture

### Data & Config Setup
Users configure paths once via:
```python
import mrna_bench as mb
mb.update_data_path("/path/to/data")
mb.update_model_weights_path("/path/to/weights")
```
Config stored in `mrna_bench/config.yaml`. Data lives at `<data_path>/<dataset_name>/` with subdirs `raw_data/`, `embeddings/`, and `data_df.parquet`.

### Module Map
```
mrna_bench/
├── models/           # EmbeddingModel subclasses (one file per model)
├── datasets/         # BenchmarkDataset subclasses (one file per dataset)
├── data_splitter/    # default / homology / kmer / chromosome splits
├── embedder/         # DatasetEmbedder — batched embedding to .npz/.h5
├── linear_probe/     # LinearProbeBuilder (builder pattern) → metrics
├── fine_tune/        # FineTuneWrapper + LoRA + TaskHead + Trainer + Persister + DataLoader
├── loader/           # load_model(), load_dataset() entry points
└── utils.py          # Config, download helpers
```

### Model Layer (`mrna_bench/models/`)

All models inherit from `EmbeddingModel` (abstract base in `embedding_model.py`). The key contract:
- `embed(sequences, cds, splice, agg_fn)` → `list[np.ndarray]`
- `extract(sequences, cds, splice, layers, return_attentions, offload_to_cpu)` → `(hidden_states, scores)`
- `get_model_short_name(model_version)` → unique string (default replaces `_` with `-`)
- `hookable_layer_patterns: ClassVar[list[str]]` — fullmatch regexes naming the modules `extract()` can hook
- Dependencies must be **lazy-imported** inside `__init__`
- Register in `MODEL_CATALOG` in `model_catalog.py`

Models can expose `lora_target_modules` (list of module name strings) for fine-tuning. Override `get_peft_target()` / `set_peft_target()` when the trainable `nn.Module` is not at `self.model` (e.g. wrappers, ensembles).

#### Attention Implementation

Each model declares two class-level attributes controlling the attention backend:
- `default_attn_implementation: ClassVar[str | None]` — used when no override is passed
- `valid_attn_implementations: ClassVar[list[str] | None]` — `None` for non-transformer models (e.g., Mamba SSMs, Hyena)

The `__init__` accepts an `attn_implementation` kwarg validated against `valid_attn_implementations` (passing a value to a model whose list is `None` raises `ValueError`). Backends per model family:

| Model family | Valid backends | Default |
|---|---|---|
| Most HF transformer models (DNABERT2/-S/-kmer, mRNABERT, RNA-FM, mRNA-FM, Plant-RNAFM, RiNALMo, RNABERT, RNAErnie, SpliceBERT, 3UTRBERT, UTR-LM, CodonBERT, Carbon, AIDO.RNA, GENERanno, GENERator, NucleotideTransformer v2, Evo1, Evo2, Helix-mRNA) | `eager`, `sdpa`, `flash_attention_2` | `flash_attention_2` |
| OmniGenome, Borzoi | `eager`, `flash_attention_2` | `flash_attention_2` |
| RNA-MSM, ERNIE-RNA, NucleotideTransformer v3, Enformer, AlphaGenome | `eager` | `eager` |
| Orthrus (Mamba), HyenaDNA (Hyena), NaiveMamba, NaiveBaseline | N/A (`None`) | `None` |

Attention weight extraction requires `eager`. With any non-eager backend (`sdpa`, `flash_attention_2`), `_standard_hf_extract` emits a warning and silently sets `return_attentions = False`.

#### Hidden State & Attention Extraction (`extract()`)

```python
hidden_states, scores = model.extract(
    sequences,                 # list[str] (or list[tuple] for multi-track models)
    cds=None,                  # list[np.ndarray] | None — required for mRNABERT/Orthrus
    splice=None,               # list[np.ndarray] | None
    layers=None,               # list[int | str] | None (None=all, int=index into
                               #   hookable_layers, str=raw get_submodule() path)
    return_attentions=False,   # only effective when attn_implementation == "eager"
    offload_to_cpu=True,       # move tensors off GPU immediately
)
# hidden_states: dict[str, list[list[torch.Tensor]]]
#   key = module path (e.g., "encoder.layer.3")
#   value[seq_idx][chunk_idx] = Tensor(T, D)
# scores: dict[str, list[list[torch.Tensor]] | None]
#   None when attention is unavailable for a layer (CNN, Mamba, flash/sdpa attn,
#   or return_attentions=False)
#   otherwise [seq_idx][chunk_idx] = Tensor(H, T, T)
```

Available layer paths are returned by the `hookable_layers` property, which by default runs `discover_layers(self.model, self.hookable_layer_patterns)` over `named_modules()`. Override the property only when layers cannot be discovered from the module tree.

**Implementation strategies across architectures:**

- **HuggingFace standard** (DNABERT family, RNA-FM, RiNALMo, NT v2, and most others): delegates to `_standard_hf_extract()`, which chunks sequences, registers hooks via `hf_extract`, and requests the model's native `output_attentions` (only honored under `eager`). A layer that never emits attention maps to `None` in `scores`.
- **Hook-only / no attention** (Orthrus, HyenaDNA, NaiveMamba): hook-based extraction on SSM/Hyena layers; `scores` values are always `None`. HyenaDNA additionally processes sequences one at a time — convolution-based Hyena operators cannot mask padding tokens, so batched variable-length inference would produce inconsistent embeddings.
- **Hybrid CNN+Transformer** (NTV3): `hookable_layer_patterns` enumerate conv-tower, transformer, and deconv-tower blocks in forward order; only transformer-block paths yield attention tensors, CNN/deconv paths return `None`.
- **Ensemble / CNN** (Borzoi, Enformer, AlphaGenome): wrap one or more backbones; `hookable_layers` may be overridden to discover layers on the underlying model(s).

### Dataset Layer (`mrna_bench/datasets/`)

All datasets inherit from `BenchmarkDataset`. Each must define:
- `METADATA: DatasetMetadata` — a frozen dataclass with `dataset_name`, `species`, `task` (`list[str]`), `target_col` (`list[str]`), `default_split_type`, `benchmark_set` (`"core"`/`"extended"`), and `vep` (bool). `task`/`target_col` are lists because a dataset can expose multiple prediction targets.
- `_get_data_from_raw()` — downloads/processes source data into `data_df.parquet`
- Rows = transcripts; required column: `sequence`; optional: `gene`, `cds`, `splice`
- Register in `DATASET_CATALOG` in `dataset_catalog.py`

Valid `task` values (in `VALID_TASKS`): `regression`, `classification`, `multilabel`, `reg_lin`, `reg_ridge`, `zeroshot`.

### Embedding Pipeline

`DatasetEmbedder` chunks datasets for parallel GPU jobs (via `d_chunk_ind`/`d_num_chunks`). Outputs:
- Pooled embeddings → `.npz` (shape: `N × hidden_dim`)
- Ragged embeddings → `.h5` (variable sequence lengths)

Call `merge_embeddings()` after all chunks complete.

### Evaluation Pipelines

**Linear Probe** (frozen embeddings → sklearn head):
```python
probe = (LinearProbeBuilder(dataset)
    .fetch_embedding_by_model_instance(model)
    .build_splitter()
    .build())
metrics = probe.run()
```

**Fine-Tuning** (LoRA + task head, end-to-end):
```python
from mrna_bench.fine_tune import (
    FineTuneWrapper, TaskHead, FineTuneTrainer, TrainerConfig, create_dataloaders
)
wrapper = FineTuneWrapper(model, task_head)
wrapper.apply_lora(rank=8, alpha=16)
train_loader, val_loader, test_loader = create_dataloaders(dataset)
config = TrainerConfig(
    learning_rate=1e-4,
    epochs=10,
    warmup_steps=100,
    early_stopping_patience=3,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
)
trainer = FineTuneTrainer(wrapper, config)
trainer.fit(train_loader, val_loader)
```

Results are saved by `FineTunePersister`; filenames encode all hyperparameters for easy aggregation.

### Key Design Patterns
- **Builder pattern:** `LinearProbeBuilder` chains configuration calls
- **Lazy imports:** every model loads its framework inside `__init__`; never at module level
- **Protocol typing:** `SupportsEmbedding`, `TaskHeadProtocol` for structural subtyping
- **Persister classes:** separate `LinearProbePersister` / `FineTunePersister` for saving results
- **Chunked embedding:** datasets split across jobs; merging happens after all chunks finish
- **Hook-based extraction:** `_register_hooks()` / `_remove_hooks()` on named modules for hidden states; hooks capture `output[0]` from tuple outputs
- **Attention gating:** `return_attentions=True` is silently disabled for non-eager backends; callers should check `scores[layer]` for `None` before use
- **CDS-aware chunking:** multi-track models (mRNABERT, Orthrus) use `chunk_sequence_cds_aware()` to keep codon frames intact across chunks
