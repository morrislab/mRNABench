# mRNABench

<div align="center">

<img width="620" alt="mRNABench logo" src="https://raw.githubusercontent.com/morrislab/mRNABench/main/assets/mrnabench-lockup-5bar-black-outlined.svg" />

[![PyPI version](https://badge.fury.io/py/mrna-bench.svg)](https://pypi.org/project/mrna-bench/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.07.05.662870-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1)

</div>

mRNABench evaluates frozen nucleotide-model representations on mature mRNA
property and function tasks. It provides a shared catalog of datasets, model
adapters, biologically informed splits, linear probes, variant-effect scoring,
and result persistence.

- **Website:** [taykhoomdalal.github.io/mRNABench](https://taykhoomdalal.github.io/mRNABench/)
- **Paper:** [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1)
- **Datasets:** [Hugging Face collection](https://huggingface.co/collections/morrislab/mrnabench-6825747c0b9253c3226078d9)
- **Notebook:** [Colab example](https://colab.research.google.com/drive/1VZF5NPwJYowAR3e6wuaiAuQyw2v7TSwx?usp=sharing)

## Install

For datasets and scikit-learn evaluation:

```bash
conda create --name mrna_bench python=3.12
conda activate mrna_bench
python -m pip install mrna-bench
```

For model inference, add PyTorch and the model dependencies:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1

python -m pip install wheel packaging ninja
python -m pip install --no-build-isolation 'mrna-bench[base_models]'
```

The model extra builds CUDA extensions for `flash-attn` and `mamba-ssm`.
Install it on a CUDA 12.6 development environment with matching headers and
`nvcc`; it is not a CPU-only installation.

Set persistent data and model-weight paths before loading resources:

```python
import mrna_bench as mb

mb.update_data_path("/absolute/path/to/mrnabench-data")
mb.update_model_weights_path("/absolute/path/to/model-weights")
```

## Quick example

```python
import torch
import mrna_bench as mb

from mrna_bench.embedder import DatasetEmbedder
from mrna_bench.linear_probe import LinearProbeBuilder

device = torch.device("cuda")
dataset = mb.load_dataset("go-mf")
model = mb.load_model(
    "Orthrus",
    "orthrus-large-6-track",
    device=device,
)

embeddings = DatasetEmbedder(
    model,
    dataset,
    batch_size=8,
).embed_dataset()
embeddings = torch.stack(embeddings).cpu().numpy()

probe = (
    LinearProbeBuilder(dataset)
    .fetch_embedding_by_embedding_instance(
        model.short_name,
        embeddings,
    )
    .build()
)

metrics = probe.run_linear_probe(random_seed=2541)
print(metrics)
```

`LinearProbeBuilder` reads the task, target, and default split from dataset
metadata. Use the website documentation for explicit split and evaluator
configuration, persisted benchmark scripts, variant-effect scoring,
fine-tuning, result analysis, and extension guides.

## Documentation

| Need | Guide |
|---|---|
| Install and run the first probe | [Quickstart](https://taykhoomdalal.github.io/mRNABench/docs/quickstart/) |
| Understand tasks, routes, and defaults | [Core concepts](https://taykhoomdalal.github.io/mRNABench/docs/concepts/) |
| Generate embeddings and run seeded probes | [Benchmarking](https://taykhoomdalal.github.io/mRNABench/docs/benchmarking/) |
| Configure splits and estimators | [Configuration](https://taykhoomdalal.github.io/mRNABench/docs/configuration/) |
| Query and interpret outputs | [Results and analysis](https://taykhoomdalal.github.io/mRNABench/docs/results/) |
| Add a dataset | [Dataset extension guide](https://taykhoomdalal.github.io/mRNABench/docs/add-dataset/) |
| Add a model | [Model extension guide](https://taykhoomdalal.github.io/mRNABench/docs/add-model/) |
| Look up Python signatures | [Python API](https://taykhoomdalal.github.io/mRNABench/docs/api/) |
| Browse supported datasets and models | [Dataset catalog](https://taykhoomdalal.github.io/mRNABench/datasets/) and [model catalog](https://taykhoomdalal.github.io/mRNABench/models/) |

## Development

```bash
conda create --name mrna_bench_dev python=3.12
conda activate mrna_bench_dev

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1

python -m pip install --no-build-isolation '.[base_models,dev]'
git config core.hooksPath .githooks
```

The tracked hook runs flake8, mypy, and CPU-only tests. Run
`./precommit.sh` for the complete suite, including model tests that require a
GPU.

## Citation

If you use mRNABench, cite bioRxiv v1 and the original sources for the datasets
and models included in your analysis.

```bibtex
@article{shi_dalal_fradkin_2025_mrnabench,
    author = {Shi, Ruian and Dalal, Taykhoom and Fradkin, Philip and Koyyalagunta, Divya and Chhabria, Simran and Jung, Andrew and Tam, Cyrus and Ceyhan, Defne and Lin, Jessica and Laverty, Kaitlin U. and Baali, Ilyes and Wang, Bo and Morris, Quaid},
    title = {mRNABench: A curated benchmark for mature mRNA property and function prediction},
    elocation-id = {2025.07.05.662870},
    year = {2025},
    doi = {10.1101/2025.07.05.662870},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/07/08/2025.07.05.662870},
    eprint = {https://www.biorxiv.org/content/early/2025/07/08/2025.07.05.662870.full.pdf},
    journal = {bioRxiv}
}
```

mRNABench is licensed under the [GNU AGPL v3](LICENSE).
