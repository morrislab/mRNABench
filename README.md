# mRNABench
This repository contains a workflow to benchmark the embedding quality of genomic foundation models on (m)RNA specific tasks. The mRNABench contains a catalogue of datasets and training split logic which can be used to evaluate the embedding quality of several catalogued models 

## Setup
The mRNA bench can be installed by cloning this repository and running:
```pip install -e .```

After installation, please call the following to set where data associated with the benchmarks will be stored.
```python
import mrna_bench as mb
mb.update_data_path(path_to_dir_to_store_data)
```

Unfortunately, there's no good way to setup each individual model at the moment. The best approach seems to be to create a conda environment, install the dependencies for the models used for embedding, and then call the above code to install mrna_bench into
each conda environment.

## Usage
Datasets can be retrieved using:

```python
import torch

import mrna_bench as mb
from mb.embedder import DatasetEmbedder
from mb.linear_probe import LinearProbe

device = torch.device("cuda")

dataset = mb.load_dataset("go-mf")
model = mb.load_model("Orthrus", "orthrus_large_6_track", device)

embedder = DatasetEmbedder(model, dataset)
embeddings = embedder.embed_dataset()

prober = LinearProbe(
    model_name="Orthrus",
    model_version="orthrus_large_6_track",
    dataset_name="go-mf",
    seq_chunk_overlap=0,
    target_col="target",
    target_task="multilabel",
    split_type="homology"
)

metrics = prober.run_linear_probe()
print(metrics)
```
Also see the `scripts/` folder for example scripts that uses slurm to embed dataset chunks in parallel for reduce runtime, as well as an example of multi-seed linear probing.

## Model Catalog
The current models catalogued are:

| Model Name |  Model Versions         | Description   | Citation |
| ---------- |  ---------------------- | --------  |  -------- |
| `Orthrus` | `orthrus_large_6_track`<br> `orthrus_base_4_track` | Mamba-based RNA FM trained using contrastive learning. | [paper](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v2)|
| `AIDO.RNA` | `aido_rna_650m` <br> `aido_rna_1b600m` <br> `aido_rna_1b600m_cds` | Encoder Transformer-based RNA FM trained using MLM on 42M ncRNA sequences. Version that is domain adapted to CDS is available. | [paper](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1) |


## Dataset Catalog
The current datasets catalogued are:
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| GO Molecular Function | `go-mf` | Classification of the molecular function of a transcript's  product as defined by the GO Resource. | `multilabel` | [website](https://geneontology.org/) |
| Mean Ribosome Load (Sugimoto) | `mrl-sugimoto` | Mean Ribosome Load per transcript isoform as measured in Sugimoto et al. 2022. | `regression` | [paper](https://www.nature.com/articles/s41594-022-00819-2) |
| RNA Half-life (Human) | `rnahl-human` | RNA half-life of human transcripts collected by Agarwal et al. 2022. | `regression` | [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) |
| RNA Half-life (Mouse) | `rnahl-mouse` | RNA half-life of mouse transcripts collected by Agarwal et al. 2022. | `regression` | [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) |
| Protein Subcellular Localization | `prot-loc` | Subcellular localization of transcript protein product defined in Protein Atlas. | `multilabel` | [website](https://www.proteinatlas.org/) |
| Protein Coding Gene Essentiality | `pcg-ess` | Essentiality of PCGs as measured by CRISPR knockdown. Log-fold expression and binary essentiality available on several cell lines. | `regression` `classification`| [paper](https://www.cell.com/cell/fulltext/S0092-8674(24)01203-0)|
