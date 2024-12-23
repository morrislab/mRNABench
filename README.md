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

## Usage
Datasets can be retrieved using:

```python
import mrna_bench as mb
from mb.embedder import DatasetEmbedder
from mb.linear_probe import LinearProbe

# NOTE: This needs to be called once after setup
mb.update_data_path(path_to_dir_to_store_data)

dataset = mb.load_dataset("go-mf")
model = md.load_model("Orthrus", "orthrus_large_6_track")

# TODO
embedder = DatasetEmbedder()
prober = LinearProbe()

metrics = prober.run()
prober.print_metrics(metrics)
```


## Model Catalog
The current models catalogued are:

| Model Name | Model Version          | Conda Env | Description   | Citation |
| ---------- | ---------------------- | --------  | ------------- | -------- |
| `AIDO.RNA` | `aido_rna_650m`        | aido      | Small AIDO.RNA model trained on ncRNA | |
|            | `aido_rna_1b600m`      |           | Large AIDO.RNA model trained on ncRNA | |
|            | `aido_rna_1b600m_cds`  |           | Large AIDO.RNA model with domain adaptation on mRNA with CDS.| |
| `Orthrus`  | `orthrus_large_6_track`| `orthrus` | Large Orthrus model | |
|            | `orthrus_base_4_track` |           | Base Orthrus model  | |


## Dataset Catalog
The current datasets catalogued are:
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| GO Molecular Function | `go-mf` | Classification of the molecular function of a transcript's  product as defined by the GO Resource. | `multilabel` |  |
| Mean Ribosome Load (Sugimoto) | `mrl_sugimoto` | Mean Ribosome Load per transcript isoform as measure in Sugimoto et al. 2022. | `regression` |  |
| RNA Half-life (Human) | `rnahl-human` | RNA half-life of human transcripts collected by Agarwal et al. 2022. | `regression` |  |
| RNA Half-life (Mouse) | `rnahl-mouse` | RNA half-life of mouse transcripts collected by Agarwal et al. 2022. | `regression` |  |
