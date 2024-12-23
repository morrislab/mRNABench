# mRNABench
This repository contains a workflow to benchmark the embedding quality of genomic foundation models on (m)RNA specific tasks.

The main functionality is to take sequences from a catalog of datasets and then embed them using a cataloged model. Datasets can be retrieved using:

```
import mrna_bench as mb
from mb.embedder import DatasetEmbedder
from mb.linear_probe import LinearProbe

dataset = mb.load_dataset("go-mf")
model = md.load_model("Orthrus", "orthrus_large_6_track")

# TODO
embedder = DatasetEmbedder()
prober = LinearProbe()

metrics = prober.run()
prober.print_metrics(metrics)
```

## Setup
The mRNA bench can be installed by cloning this repository and running:
```pip install -e .```

## Model Catalog
The current models cataloged are:

| Model Name | Model Version          | Conda Env | Description   | Citation |
| ---------- | ---------------------- | --------  | ------------- | -------- |
| `AIDO.RNA` | `aido_rna_650m`        | aido      | Small AIDO.RNA model trained on ncRNA | |
|            | `aido_rna_1b600m`      |           | Large AIDO.RNA model trained on ncRNA | |
|            | `aido_rna_1b600m_cds`  |           | Large AIDO.RNA model with domain adaptation on mRNA with CDS.| |
| `Orthrus`  | `orthrus_large_6_track`| `orthrus` | Large Orthrus model | |
|            | `orthrus_base_4_track` |           | Base Orthrus model  | |


## Dataset Catalog
The current datasets cataloged are:
| Dataset Name | Catalog Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| GO Molecular Function | `go-mf` | Classification of the molecular function of a transcript's  product as defined by the GO Resource. | `multilabel` |  |
| Mean Ribosome Load (Sugimoto) | `mrl_sugimoto` | Mean Ribosome Load per transcript isoform as measure in Sugimoto et al. 2022. | `regression` |  |
| RNA Half-life (Human) | `rnahl-human` | RNA half-life of human transcripts collected by Agarwal et al. 2022. | `regression` |  |
| RNA Half-life (Mouse) | `rnahl-mouse` | RNA half-life of mouse transcripts collected by Agarwal et al. 2022. | `regression` |  |
