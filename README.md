# mRNABench

<div align="center">
    
[![PyPI version](https://badge.fury.io/py/mrna-bench.svg)](https://badge.fury.io/py/mrna-bench)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.07.05.662870-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1)

<img width="650" height="466" alt="image" center src="https://github.com/user-attachments/assets/f43be914-d6e7-4a71-8dda-146cc09a6c05" />

</div>

This repository contains the code for mRNABench, which benchmarks the embedding quality of genomic foundation models on mRNA specific tasks. The mRNABench contains a catalogue of datasets and training split logic which can be used to evaluate the embedding quality of several catalogued models.

**Paper:** [BioRxiv Link](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1)  
**Notebook Example:** [Colab Notebook](https://colab.research.google.com/drive/1VZF5NPwJYowAR3e6wuaiAuQyw2v7TSwx?usp=sharing)  
**Dataset Repository:** [HuggingFace Collection](https://huggingface.co/collections/morrislab/mrnabench-6825747c0b9253c3226078d9)

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Model Catalog](#model-catalog)
- [Dataset Catalog](#dataset-catalog)
- [Citation](#citation)

## Setup
Several configurations of the mRNABench are available.

### Datasets Only
If you are interested in the benchmark datasets **only**, you can run:

```bash
pip install mrna-bench
```

### Base Models
> [!IMPORTANT]  
> **Requirements:** PyTorch 2.2.2 and CUDA 12.1+ are required for base models installation.

The inference-capable version of mRNABench that can generate embeddings using most models (except Evo2 and Helix mRNA) can be installed as shown below.

```bash
conda create --name mrna_bench python=3.10
conda activate mrna_bench

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install mrna-bench[base_models]
```
Inference with other models will require the installation of the model's
dependencies first, which are usually listed on the model's GitHub page (see below).

### Post-install
> [!IMPORTANT]
> After installation, please run the following in Python to set where data associated with the benchmarks will be stored.
```python
import mrna_bench as mb

path_to_dir_to_store_data = "DESIRED_PATH"
mb.update_data_path(path_to_dir_to_store_data)

path_to_dir_to_store_weights = "DESIRED_PATH_FOR_MODEL_WEIGHTS"
mb.update_model_weights_path(path_to_dir_to_store_weights)
```

### Evo2
Evo2 requires more complicated installation instructions, and can only be run on H100s. See [Evo2 Setup](#evo2-setup).

### Dev Mode
Dev mode allows generation of datasets from scratch and includes access to the RNA-Fazal localization dataset. See [Dev Mode Setup](#dev-mode-setup).


## Usage
Datasets can be retrieved using:

```python
import mrna_bench as mb

dataset = mb.load_dataset("go-mf")
data_df = dataset.data_df
```

The mRNABench can also be used to test out common genomic foundation models:
```python
import torch

import mrna_bench as mb
from mrna_bench.embedder import DatasetEmbedder
from mrna_bench.linear_probe import LinearProbeBuilder

device = torch.device("cuda")

dataset = mb.load_dataset("go-mf")
model = mb.load_model("Orthrus", "orthrus-large-6-track", device)

embedder = DatasetEmbedder(model, dataset)
embeddings = embedder.embed_dataset()
embeddings = embeddings.detach().cpu().numpy()

prober = (LinearProbeBuilder(dataset)
    .fetch_embedding_by_embedding_instance("orthrus-large-6", embeddings)
    .build_splitter("homology", species="human", eval_all_splits=False)
    .build_evaluator("multilabel")
    .set_target("target")
    .build()
)

metrics = prober.run_linear_probe(2541)
print(metrics)
```
Also see the `scripts/` folder for example scripts that uses slurm to embed dataset chunks in parallel for reduce runtime, as well as an example of multi-seed linear probing.

## Model Catalog
The models supported by the `base_models` installation are catalogued below.

### RNA Foundation Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **Orthrus** | `orthrus-large-6-track`<br>`orthrus-base-4-track` | Mamba-based RNA foundation model pre-trained using contrastive learning on 45M RNA transcripts to capture functional and evolutionary relationships. 6-track version incorporates CDS and splice site information. | [[Code]](https://github.com/bowang-lab/Orthrus) [[Paper]](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v2)|
| **RNA-FM** | `rna-fm`<br>`mrna-fm` | Transformer-based RNA foundation model pre-trained using MLM. RNA-FM trained on 23M ncRNA sequences, mRNA-FM trained on mRNA CDS regions using codon tokenizer. | [[Github]](https://github.com/ml4bio/RNA-FM) |
| **SpliceBERT** | `SpliceBERT.1024nt`<br>`SpliceBERT-human.510nt`<br>`SpliceBERT.510nt` | Transformer-based RNA foundation model trained on 2M vertebrate mRNA sequences using MLM. Specialized for splice site prediction with human-only and context-length variants. | [[Github]](https://github.com/chenkenbio/SpliceBERT) |
| **RiNALMo** | `rinalmo` | Transformer-based RNA foundation model trained on 36M ncRNA sequences using MLM with modern architectural improvements including RoPE, SwiGLU activations, and Flash Attention. | [[Github]](https://github.com/lbcb-sci/RiNALMo) |
| **UTR-LM** | `utrlm-te_el`<br>`utrlm-mrl`<br>`utrlm-*-utronly` | Transformer-based RNA foundation model specialized for 5'UTR sequences. Pre-trained on random and endogenous UTR sequences from various species. UTR-only variants automatically extract 5'UTRs. | [[Github]](https://github.com/a96123155/UTR-LM) |
| **3UTRBERT** | `utrbert-3mer`<br>`utrbert-4mer`<br>`utrbert-5mer`<br>`utrbert-6mer`<br>`utrbert-*-utronly` | Transformer-based RNA foundation model specialized for 3'UTR regions. Uses k-mer tokenization (3-6mers) and trained on 100k 3'UTR sequences. UTR-only variants automatically extract 3'UTRs. | [[Github]](https://github.com/yangyn533/3UTRBERT) |
| **RNA-MSM** | `rnamsm` | Structure-aware RNA foundation model trained using multiple sequence alignments from custom structure-based homology mapping across ~4000 RNA families. | [[Github]](https://github.com/yikunpku/RNA-MSM) |
| **RNAErnie** | `rnaernie` | Transformer-based RNA foundation model trained using MLM with motif-level masking strategy on 23M ncRNA sequences. Uses contiguous token masking to learn RNA motifs. | [[Github]](https://github.com/CatIIIIIIII/RNAErnie) |
| **ERNIE-RNA** | `ernierna`<br>`ernierna-ss` | Transformer-based RNA foundation model with structural attention bias. Trained on 20M ncRNA sequences with custom attention incorporating RNA base pairing rules. SS version fine-tuned on structural tasks. | [[Github]](https://github.com/Bruce-ywj/ERNIE-RNA) |
| **RNABERT** | `rnabert` | Transformer-based RNA foundation model with dual training objectives combining MLM and structural alignment learning. Trained on 80k ncRNA sequences. | [[Github]](https://github.com/mana438/RNABERT) |
| **CodonBERT** | `codonbert` | Transformer-based RNA foundation model trained on 10M+ mRNA sequences from mammals, bacteria, and viruses. Specialized for coding regions and mRNA properties. | [[Github]](https://github.com/Sanofi-Public/CodonBERT) |
| **Helix-mRNA** | `helix-mrna` | Hybrid Mamba2/Transformer model trained on 26M diverse eukaryotic and viral mRNAs. Features CDS-aware tokenization with special tokens at codon boundaries. | [[Github]](https://github.com/helicalAI/helical) |

### DNA Foundation Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **DNABERT2** | `dnabert2` | Modern Transformer-based DNA foundation model with BPE tokenization and rotary positional encoding. Pre-trained using MLM on multi-species genomic datasets. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_2) |
| **DNABERT-S** | `dnabert-s` | Species-aware DNA foundation model trained with contrastive learning to encourage species grouping while discouraging cross-species associations. Covers microbial genomes including viruses, fungi, and bacteria. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_S) |
| **Nucleotide Transformer** | `2.5b-multi-species`<br>`2.5b-1000g`<br>`500m-human-ref`<br>`500m-1000g`<br>`v2-50m-multi-species`<br>`v2-100m-multi-species`<br>`v2-250m-multi-species`<br>`v2-500m-multi-species` | Transformer-based DNA foundation model family with 6-mer tokenization. Available in multiple sizes (50M-2.5B parameters) trained on various genomic datasets from human reference to multi-species collections. | [[Github]](https://github.com/instadeepai/nucleotide-transformer) |
| **HyenaDNA** | `hyenadna-large-1m-seqlen-hf`<br>`hyenadna-medium-450k-seqlen-hf`<br>`hyenadna-medium-160k-seqlen-hf`<br>`hyenadna-small-32k-seqlen-hf`<br>`hyenadna-tiny-16k-seqlen-d128-hf` | Hyena-based DNA foundation model with near-linear scaling and ultra-long context capability. Pre-trained using next token prediction on human reference genome with various model sizes and sequence lengths. | [[Github]](https://github.com/HazyResearch/hyena-dna) |
| **Evo1** | `evo-1.5-8k-base`<br>`evo-1-8k-base`<br>`evo-1-131k-base` | StripedHyena-based DNA foundation model trained autoregressively on OpenGenome dataset at single nucleotide, byte-level resolution. Offers near-linear scaling with ultra-long context variants up to 131k nucleotides. | [[Github]](https://github.com/evo-design/evo) |
| **Evo2** | `evo2_40b`<br>`evo2_40b_base`<br>`evo2_7b`<br>`evo2_7b_base`<br>`evo2_1b_base` | Next-generation StripedHyena2-based DNA foundation model trained on OpenGenome2 dataset. Provides multi-layer embeddings with ultra-long context capability up to 1M nucleotides for large model variants. | [[Github]](https://github.com/evo-design/evo2) |


### Baseline Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **NaiveBaseline** | `naive-4-track`<br>`naive-6-track` | Non-neural baseline using traditional sequence features including k-mer counts (3-7mers), GC content, and sequence statistics. 6-track version adds CDS length and exon count from structural annotations. | N/A |
| **NaiveMamba** | `naive-mamba` | Randomly initialized Mamba model serving as an untrained baseline. Uses 6-track input (sequence + CDS + splice information) with fixed random seed for reproducible comparisons. | N/A |

> [!NOTE]
> Many of the model wrappers (3UTRBERT, RiNALMo, UTR-LM, RNA-MSM, RNAErnie) use reimplementations from the `multimolecule` package. See their [website](https://multimolecule.danling.org/) for more details.

### Adding a new model
All models should inherit from the template `EmbeddingModel`. Each model file should lazily load dependencies within its `__init__` methods so each model can be used individually without install all other models. Models must implement `get_model_short_name(model_version)` which fetches the internal name for the model. This must be unique for every model version and must not contain underscores. Models should implement either `embed_sequence` or `embed_sequence_sixtrack` (see code for method signature). New models should be added to `MODEL_CATALOG`.

## Dataset Catalog
The current datasets catalogued are:

### Gene Function Annotation
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| GO Molecular Function | <code>go-mf</code> | Classification of the molecular function of a transcript's product as defined by the GO Resource. | `multilabel` | [website](https://geneontology.org/) |
| GO Biological Process | <code>go-bp</code> | Classification of the biological process a transcript's product participates in as defined by the GO Resource. | `multilabel` | [website](https://geneontology.org/) |
| GO Cellular Component | <code>go-cc</code> | Classification of the cellular component where a transcript's product is localized as defined by the GO Resource. | `multilabel` | [website](https://geneontology.org/) |

### Translation Regulation
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| Mean Ribosome Load (Sugimoto) | <code>mrl&#8209;sugimoto</code> | Mean ribosome load (MRL) per transcript isoform as measured in human cells using isoform-resolved ribosome profiling. | `regression` | [paper](https://www.nature.com/articles/s41594-022-00819-2) |
| Mean Ribosome Load (Sample) | <code>mrl&#8209;sample&#8209;egfp</code> <br><code>mrl&#8209;sample&#8209;mcherry</code><br><code>mrl&#8209;sample&#8209;designed</code><br><code>mrl&#8209;sample&#8209;varying</code> | Mean ribosome load (MRL) measured in an MPRA of randomized and designed 5'UTR regions attached to eGFP or mCherry reporters. Includes various RNA modifications and UTR lengths. | `regression` | [paper](https://pubmed.ncbi.nlm.nih.gov/31267113/)|
| Mean Ribosome Load & Half-life | <code>mrl&#8209;hl&#8209;lbkwk</code> | Joint prediction of ribosome load and RNA half-life from synthetic mRNA sequences in the Leppek et al. dataset. | `regression` | [paper](https://pubmed.ncbi.nlm.nih.gov/33821271/) |

### RNA Stability
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| RNA Half-life (Human) | <code>rnahl&#8209;human</code> | RNA half-life of human transcripts measured using time-course RNA-seq after transcription inhibition. | `regression` | [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) |
| RNA Half-life (Mouse) | <code>rnahl&#8209;mouse</code> | RNA half-life of mouse transcripts measured using time-course RNA-seq after transcription inhibition. | `regression` | [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) |

### Protein-RNA Interactions
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| eCLIP RBP Binding (K562) | <code>eclip&#8209;binding&#8209;k562</code> | RNA-binding protein (RBP) binding sites on mRNA sequences identified using eCLIP-seq in K562 cells. Covers ~80 different RBPs. | `multilabel` | [paper](https://www.nature.com/articles/s41586-020-2077-3) |
| eCLIP RBP Binding (HepG2) | <code>eclip&#8209;binding&#8209;hepg2</code> | RNA-binding protein (RBP) binding sites on mRNA sequences identified using eCLIP-seq in HepG2 cells. Covers ~70 different RBPs. | `multilabel` | [paper](https://www.nature.com/articles/s41586-020-2077-3) |

### Subcellular Localization
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| Protein Subcellular Localization | <code>prot&#8209;loc</code> | Subcellular localization of transcript protein products based on experimental evidence from the Human Protein Atlas. | `multilabel` | [website](https://www.proteinatlas.org/) |
| RNA Subcellular Localization (Fazal) | <code>rna&#8209;loc&#8209;fazal</code> | Subcellular localization of mRNA molecules measured using APEX-seq (proximity labeling + RNA-seq) in human cells. | `multilabel` | [paper](https://doi.org/10.1016/j.cell.2019.05.027) |
| RNA Subcellular Localization (Ietswaart) | <code>rna&#8209;loc&#8209;ietswaart</code> | Subcellular localization of mRNA molecules in human cells using compartment-specific RNA-seq approaches. | `multilabel` | [paper](https://pubmed.ncbi.nlm.nih.gov/38964322/) |

### Variant Effect Prediction
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| VEP TraitGym (Mendelian) | <code>vep&#8209;traitgym&#8209;mendelian</code> | Pathogenicity prediction for genetic variants in 3'UTR and 5'UTR regions associated with Mendelian diseases. | `classification` | [paper](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) |
| VEP TraitGym (Complex) | <code>vep&#8209;traitgym&#8209;complex</code> | Pathogenicity prediction for genetic variants in 3'UTR and 5'UTR regions associated with complex traits. | `classification` | [paper](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) |

### Adding a new dataset
New datasets should inherit from `BenchmarkDataset`. Dataset names cannot contain underscores. Each new dataset should download raw data and process it into a dataframe by overriding `process_raw_data`. This dataframe should store transcript as rows, using string encoding in the `sequence` column. If homology splitting is required, a column `gene` containing gene names is required. Six track embedding also requires columns `cds` and `splice`. The target column can have any name, as it is specified at time of probing. New datasets should be added to `DATASET_CATALOG`.

## Citation
If you use mRNABench in your research, please cite:

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

## Evo2 Setup
Inference using Evo2 requires installing the following in its own environment. Note: There may be an issue where the evo_40b models, when downloaded, have their merged checkpoints stored one directory above the HuggingFace hub cache. You may need to manually move the checkpoint into its corresponding snapshot directory: `/hub/models--arcinstitute-evo2_40b*/snapshots/snapshot_name/`

**Hardware Requirements:** Evo2 can only be run on H100 GPUs.

```bash
conda create --name evo_bench -c conda-forge python=3.11 gxx=12.2.0 -y
conda activate evo_bench

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install vtx==1.0.4
pip install evo2==0.2.0
pip install flash-attn==2.7.4.post1

cd path/to/mRNA/bench
pip install -e .
```

## Dev Mode Setup
Dev mode requires additional dependencies for generating datasets from scratch and accessing the RNA-Fazal localization dataset. 

```bash
conda create --name mrna_bench_dev python=3.10
conda activate mrna_bench_dev

# Install genome-kit first
conda install -c conda-forge genome_kit=7.1.0
conda install -c conda-forge gcc_linux-64 gxx_linux-64
# Note: You might need to add gcc compilers to LD_LIBRARY_PATH if you encounter linking issues

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install mrna-bench[base_models]
```
