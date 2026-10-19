# mRNABench

<div align="center">

[![PyPI version](https://badge.fury.io/py/mrna-bench.svg)](https://badge.fury.io/py/mrna-bench)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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
> **Requirements:** PyTorch 2.7.1 with CUDA 12.6 is required for base models installation.

The inference-capable version of mRNABench can generate embeddings using all catalogued models including Evo2, Orthrus, and MultiMolecule models.

```bash
conda create --name mrna_bench python=3.12
conda activate mrna_bench

pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
pip install psutil
pip install --no-build-isolation -e .[base_models]
```

flash-attn and mamba-ssm build against the installed torch, so they need
`--no-build-isolation`, and the flash-attn build imports psutil. For Evo2 on
GPUs with FP8 support, see [Evo2 Setup](#evo2-setup).

### Post-install (Optional)
By default, benchmark data is stored in `~/.cache/mrna_bench` and model weights in `~/.cache/mrna_bench/model_weights`. To use a custom location, run:
```python
import mrna_bench as mb

mb.update_data_path("DESIRED_PATH")
mb.update_model_weights_path("DESIRED_PATH_FOR_MODEL_WEIGHTS")
```

### Evo2
Evo2 is included in the base_models installation and runs on any GPU with enough memory. See [Evo2 Setup](#evo2-setup) for FP8 and memory requirements.

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
    .build()
)

metrics = prober.run_linear_probe(2541)
print(metrics)
```
`LinearProbeBuilder` uses dataset metadata defaults for task, target,
and split strategy. You only need to set these explicitly when you want to
override defaults, for example:

```python
prober = (LinearProbeBuilder(dataset)
    .fetch_embedding_by_embedding_instance("orthrus-large-6", embeddings)
    .build_splitter("homology", species="human", eval_all_splits=True)
    .build_evaluator("multilabel")
    .set_target("target")
    .build()
)
```

Calling `.use_persister()` on the builder saves results to a SQLite database at
`{dataset_path}/results.db`; re-running the same configuration overwrites its row.

Variant effect datasets (`vep-traitgym-*`, `utr-variants-bohn-*`) are probed on
the difference between each variant's embedding and its wild-type embedding.
They also support zero-shot scoring, which ranks variants by the L2 norm of
that difference and reports AUROC and AUPRC:

```python
dataset = mb.load_dataset("vep-traitgym-mendelian")
embeddings = DatasetEmbedder(model, dataset).embed_dataset().cpu().numpy()

zero_shot = (LinearProbeBuilder(dataset)
    .fetch_embedding_by_embedding_instance("orthrus-large-6", embeddings)
    .build_evaluator("zeroshot")
    .build()
)
print(zero_shot.run())
```

Linear probing results from the paper, for all split types, seeds, models,
datasets and tasks, are in `paper/leaderboard_results.parquet`.

Models embed batches of sequences directly. `load_model` returns a model in
inference mode; call `set_train_mode()` to enable gradients through the
backbone, e.g. to train a head on top:

```python
model = mb.load_model("RiNALMo")  # default version and device
model.set_train_mode()
head = torch.nn.LazyLinear(1).to(model.device)

embeddings = model.embed(["ACGTACGT", "ATGCCGTAATG"])  # (batch_size, H)
loss = head(embeddings).sum()
loss.backward()  # gradients reach the backbone weights

model.set_inference_mode()
```

Also see the `scripts/` folder for example scripts that uses slurm to embed dataset chunks in parallel for reduce runtime, as well as an example of multi-seed linear probing.

## Model Catalog
The models supported by the `base_models` installation are catalogued below.

### RNA Foundation Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **Orthrus** | `orthrus-large-6-track`<br>`orthrus-large-4-track`<br>`orthrus-base-4-track` | Mamba-based RNA foundation model pre-trained using contrastive learning on 45M RNA transcripts to capture functional and evolutionary relationships. 6-track version incorporates CDS and splice site information. | [[Code]](https://github.com/bowang-lab/Orthrus) [[Paper]](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v2)|
| **RNA-FM** | `rna-fm` | Transformer-based RNA foundation model pre-trained using MLM on 23M ncRNA sequences. | [[Github]](https://github.com/ml4bio/RNA-FM) |
| **mRNA-FM** | `mrna-fm` | RNA-FM variant pre-trained on mRNA CDS regions using a codon tokenizer. Accepts CDS input only (length a multiple of 3). | [[Github]](https://github.com/ml4bio/RNA-FM) |
| **SpliceBERT** | `SpliceBERT.1024nt`<br>`SpliceBERT-human.510nt`<br>`SpliceBERT.510nt` | Transformer-based RNA foundation model trained on 2M vertebrate mRNA sequences using MLM. Specialized for splice site prediction with human-only and context-length variants. | [[Github]](https://github.com/chenkenbio/SpliceBERT) |
| **RiNALMo** | `rinalmo-giga`<br>`rinalmo-mega`<br>`rinalmo-micro` | Transformer-based RNA foundation model trained on 36M ncRNA sequences using MLM with modern architectural improvements including RoPE, SwiGLU activations, and Flash Attention. | [[Github]](https://github.com/lbcb-sci/RiNALMo) |
| **UTR-LM** | `utrlm-te_el`<br>`utrlm-mrl` | Transformer-based RNA foundation model specialized for 5'UTR sequences. Pre-trained on random and endogenous UTR sequences from various species. | [[Github]](https://github.com/a96123155/UTR-LM) |
| **3UTRBERT** | `utrbert-3mer`<br>`utrbert-4mer`<br>`utrbert-5mer`<br>`utrbert-6mer` | Transformer-based RNA foundation model specialized for 3'UTR regions. Uses k-mer tokenization (3-6mers) and trained on 100k 3'UTR sequences. | [[Github]](https://github.com/yangyn533/3UTRBERT) |
| **RNA-MSM** | `rnamsm` | Structure-aware RNA foundation model trained using multiple sequence alignments from custom structure-based homology mapping across ~4000 RNA families. | [[Github]](https://github.com/yikunpku/RNA-MSM) |
| **RNAErnie** | `rnaernie` | Transformer-based RNA foundation model trained using MLM with motif-level masking strategy on 23M ncRNA sequences. Uses contiguous token masking to learn RNA motifs. | [[Github]](https://github.com/CatIIIIIIII/RNAErnie) |
| **ERNIE-RNA** | `ernierna`<br>`ernierna-ss` | Transformer-based RNA foundation model with structural attention bias. Trained on 20M ncRNA sequences with custom attention incorporating RNA base pairing rules. SS version fine-tuned on structural tasks. | [[Github]](https://github.com/Bruce-ywj/ERNIE-RNA) |
| **RNABERT** | `rnabert` | Transformer-based RNA foundation model with dual training objectives combining MLM and structural alignment learning. Trained on 80k ncRNA sequences. | [[Github]](https://github.com/mana438/RNABERT) |
| **CodonBERT** | `codonbert` | Transformer-based RNA foundation model trained on 10M+ mRNA sequences from mammals, bacteria, and viruses. Specialized for coding regions and mRNA properties. | [[Github]](https://github.com/Sanofi-Public/CodonBERT) |
| **Helix-mRNA** | `helix-mrna` | Hybrid Mamba2/Transformer model trained on 26M diverse eukaryotic and viral mRNAs. Features CDS-aware tokenization with special tokens at codon boundaries. | [[Github]](https://github.com/helicalAI/helical) |
| **mRNABERT** | `mRNABERT` | Transformer-based RNA foundation model with ALiBi positional embeddings, pre-trained using MLM on 36M mRNA sequences and further trained with contrastive learning to align CDS embeddings with ProtT5 protein embeddings. | [[Github]](https://github.com/yyly6/mRNABERT) |
| **Plant-RNAFM** | `plant_rnafm` | Transformer-based RNA foundation model pre-trained on 25M RNA sequences from 1,124 plant species using MLM, secondary structure prediction, and RNA region annotation objectives. | [[Github]](https://github.com/yangheng95/PlantRNA-FM) |
| **OmniGenome** | `omnigenome-52m`<br>`omnigenome-186m` | Structure-aware transformer-based RNA foundation model pre-trained on plant RNA sequences from the OneKP initiative with sequence-to-structure, structure-to-sequence, and MLM objectives. | [[Github]](https://github.com/yangheng95/OmniGenBench) |

### DNA Foundation Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **AlphaGenome** | `alphagenome` | Convolutional and Transformer model predicting functional genomic tracks from up to 1 Mb of DNA. Uses the PyTorch port of the DeepMind weights, which are subject to the [AlphaGenome Model Terms](https://deepmind.google.com/science/alphagenome/model-terms). | [[Github]](https://github.com/google-deepmind/alphagenome_research) |
| **Borzoi** | `borzoi`<br>`flashzoi`<br>`borzoi-replicate-{0-3}`<br>`flashzoi-replicate-{0-3}` | Convolutional and Transformer model predicting RNA-seq coverage in 32 bp bins from 524 kb of DNA. `borzoi` and `flashzoi` average the four replicates; Flashzoi is a variant that uses FlashAttention-2 for faster inference. | [[Github]](https://github.com/johahi/borzoi-pytorch) [[Paper]](https://www.nature.com/articles/s41588-024-02053-6) |
| **Enformer** | `enformer-official-rough` | Convolutional and Transformer model predicting functional genomic tracks from 200 kb of DNA. Uses the PyTorch port of the DeepMind weights. | [[Github]](https://github.com/lucidrains/enformer-pytorch) [[Paper]](https://www.nature.com/articles/s41592-021-01252-x) |
| **DNABERT2** | `dnabert2` | Modern Transformer-based DNA foundation model with BPE tokenization and rotary positional encoding. Pre-trained using MLM on multi-species genomic datasets. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_2) |
| **DNABERT-S** | `dnabert-s` | Species-aware DNA foundation model trained with contrastive learning to encourage species grouping while discouraging cross-species associations. Covers microbial genomes including viruses, fungi, and bacteria. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_S) |
| **NucleotideTransformer** | `2.5b-multi-species`<br>`2.5b-1000g`<br>`500m-human-ref`<br>`500m-1000g`<br>`v2-50m-multi-species`<br>`v2-100m-multi-species`<br>`v2-250m-multi-species`<br>`v2-500m-multi-species` | Transformer-based DNA foundation model family with 6-mer tokenization. Available in multiple sizes (50M-2.5B parameters) trained on various genomic datasets from human reference to multi-species collections. | [[Github]](https://github.com/instadeepai/nucleotide-transformer) |
| **NucleotideTransformerV3** | `v3_8M_pre`<br>`v3_100M_pre`<br>`v3_650M_pre`<br>`v3_100M_post`<br>`v3_650M_post` | U-Net style DNA foundation model (convolutional down/upsampling around a Transformer tower) at single-nucleotide resolution, trained on OpenGenome2 with sequences up to 1 Mb. Post-trained variants are further trained to predict 16K+ genomic tracks. | [[Github]](https://github.com/instadeepai/nucleotide-transformer) |
| **HyenaDNA** | `hyenadna-large-1m-seqlen-hf`<br>`hyenadna-medium-450k-seqlen-hf`<br>`hyenadna-medium-160k-seqlen-hf`<br>`hyenadna-small-32k-seqlen-hf`<br>`hyenadna-tiny-16k-seqlen-d128-hf` | Hyena-based DNA foundation model with near-linear scaling and ultra-long context capability. Pre-trained using next token prediction on human reference genome with various model sizes and sequence lengths. | [[Github]](https://github.com/HazyResearch/hyena-dna) |
| **Evo1** | `evo-1.5-8k-base`<br>`evo-1-8k-base`<br>`evo-1-131k-base` | StripedHyena-based DNA foundation model trained autoregressively on OpenGenome dataset at single nucleotide, byte-level resolution. Offers near-linear scaling with ultra-long context variants up to 131k nucleotides. | [[Github]](https://github.com/evo-design/evo) |
| **Evo2** | `evo2_40b`<br>`evo2_40b_base`<br>`evo2_20b`<br>`evo2_7b`<br>`evo2_7b_262k`<br>`evo2_7b_base`<br>`evo2_1b_base` | Next-generation StripedHyena2-based DNA foundation model trained on OpenGenome2 dataset. Provides multi-layer embeddings with ultra-long context capability up to 1M nucleotides for large model variants. | [[Github]](https://github.com/evo-design/evo2) |
| **GENERanno** | `eukaryote-0.5b-base`<br>`eukaryote-1.2b-cds-annotator-preview`<br>`prokaryote-0.5b-base`<br>`prokaryote-0.5b-cds-annotator` | Transformer-encoder genomic foundation model at single-nucleotide resolution with bidirectional attention over up to 8k bp. CDS-annotator checkpoints are finetuned for metagenomic CDS calling. | [[Github]](https://github.com/GenerTeam/GENERanno) |
| **GENERator** | `eukaryote-1.2b-base`<br>`eukaryote-3b-base`<br>`v2-eukaryote-1.2b-base`<br>`v2-eukaryote-3b-base`<br>`v2-prokaryote-1.2b-base`<br>`v2-prokaryote-3b-base` | Autoregressive Transformer genomic foundation model with k-mer tokenization, trained on gene-centric functional regions. v2 adds nucleotide-level supervision and supports contexts up to 98k bp. | [[Github]](https://github.com/GenerTeam/GENERator) |


### Baseline Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **NaiveBaseline** | `naive-4-track` | Non-neural baseline using k-mer counts (3-7mers) and GC content. | N/A |
| **NaiveBaselineSixTrack** | `naive-6-track` | NaiveBaseline features plus CDS length and exon count from structural annotations. | N/A |
| **NaiveMamba** | `naive-mamba` | Randomly initialized Mamba model serving as an untrained baseline. Uses 6-track input (sequence + CDS + splice information) with fixed random seed for reproducible comparisons. | N/A |

> [!NOTE]
> Many of the model wrappers (3UTRBERT, ERNIE-RNA, RiNALMo, RNABERT, UTR-LM, RNA-MSM, RNAErnie) use reimplementations from the `multimolecule` package. See their [website](https://multimolecule.danling.org/) for more details.

### Adding a new model
All models should inherit from the template `EmbeddingModel`. Each model file should lazily load dependencies within its `__init__` methods so each model can be used individually without install all other models. Models must define `default_version` and `valid_versions`, implement `get_model_short_name(model_version)`, which returns the internal name for the model (unique for every model version, no underscores), and implement `embed(sequences, cds, splice, agg_fn)`, which embeds a batch of sequences into a `(batch_size, H)` tensor without blocking gradients (see code for method signature). New models should be added to `MODEL_CATALOG`.

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
| Translation Efficiency (Human) | <code>translation&#8209;efficiency&#8209;human</code> | Translation efficiency of human transcripts measured using ribosome profiling. | `regression` | [paper](https://www.nature.com/articles/s41587-025-02712-x) |
| Translation Efficiency (Mouse) | <code>translation&#8209;efficiency&#8209;mouse</code> | Translation efficiency of mouse transcripts measured using ribosome profiling. | `regression` | [paper](https://www.nature.com/articles/s41587-025-02712-x) |
| IRES Classification | <code>ires&#8209;classification</code> | Classification of internal ribosome entry site activity across assayed and curated candidate sequences. | `classification` | [paper](https://doi.org/10.1038/s42256-024-00823-9) |

### Alternative Polyadenylation
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| APA Isoform | <code>apa&#8209;isoform</code> | Proximal isoform usage measured across synthetic 3' UTR APA windows from the APARENT libraries distributed by BEACON. | `regression` | [paper](https://doi.org/10.1016/j.cell.2019.04.046) |

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

### RNA Lifecycle
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| RNA Lifecycle (Ietswaart) | <code>rna&#8209;lifecycle&#8209;ietswaart</code> | RNA lifecycle properties including synthesis, processing, and degradation rates in human cells. | `multilabel` | [paper](https://pubmed.ncbi.nlm.nih.gov/38964322/) |

### miRNA Target Prediction
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| miRNA Target (MirTarClash) | <code>mirna&#8209;target</code> | Experimentally validated miRNA target sites on human mRNAs from CLASH-based experiments. Binary classification for top 20 most frequent miRNAs. | `multilabel` | [paper](https://academic.oup.com/database/article/doi/10.1093/database/baaf023/8106627) |

### Variant Effect Prediction
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| VEP TraitGym (Mendelian) | <code>vep&#8209;traitgym&#8209;mendelian</code> | Pathogenicity prediction for genetic variants in 3'UTR and 5'UTR regions associated with Mendelian diseases. | `classification` | [paper](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) |
| VEP TraitGym (Complex) | <code>vep&#8209;traitgym&#8209;complex</code> | Pathogenicity prediction for genetic variants in 3'UTR and 5'UTR regions associated with complex traits. | `classification` | [paper](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) |
| UTR Variants (Bohn) | <code>utr&#8209;variants&#8209;bohn&#8209;utr5</code><br><code>utr&#8209;variants&#8209;bohn&#8209;utr3</code> | Variant effect prediction for 5'UTR and 3'UTR variants from Bohn et al. | `classification` | [paper](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2023.1257550/full) |

### Adding a new dataset
New datasets should inherit from `BenchmarkDataset`. Dataset names cannot contain underscores. Each new dataset should download raw data and process it into a dataframe by overriding `_get_data_from_raw`. This dataframe should store transcript as rows, using string encoding in the `sequence` column. If homology splitting is required, a column `gene` containing gene names is required. Six track embedding also requires columns `cds` and `splice`. The target column can have any name, as it is specified at time of probing. New datasets should be added to `DATASET_CATALOG`.

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

The original sources for each dataset and model should be cited if used, and can be found above. A substantial number of model implementations use the the `multimolecule` package: https://huggingface.co/multimolecule; citation information can be found on their HuggingFace.


## Evo2 Setup
Evo1 and Evo2 load from HuggingFace ports of the original checkpoints (`Taykhoom/Evo1-*`, `Taykhoom/Evo2-*`), pinned to fixed revisions.

**FP8:** Evo2 checkpoints use FP8 input projections, which need TransformerEngine and a GPU with compute capability 8.9+ (e.g. H100). On other GPUs, or without TransformerEngine, the projections run in bf16 with a warning, and embeddings differ from the FP8 reference. To use FP8, install the `fp8` extra; TransformerEngine builds against torch's cuDNN headers:

```bash
CUDNN=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")
CUDNN_PATH=$CUDNN CPATH=$CUDNN/include LIBRARY_PATH=$CUDNN/lib \
    pip install --no-build-isolation -e .[fp8]
```

Set the cuDNN variables only for this command: TransformerEngine fails to import if `CUDNN_PATH` stays set.

**Memory (bf16 weights):** `evo2_1b_base` ~2 GB; 7B variants ~14 GB; `evo2_20b` ~39 GB; 40B variants ~82 GB.

## Dev Mode Setup
Dev mode requires additional dependencies for generating datasets from scratch and accessing certain datasets.

```bash
conda create --name mrna_bench_dev python=3.12
conda activate mrna_bench_dev

pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
pip install psutil
pip install --no-build-isolation -e .[base_models,dev]
```
