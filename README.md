# mRNABench

<div align="center">

[![PyPI version](https://badge.fury.io/py/mrna-bench.svg)](https://badge.fury.io/py/mrna-bench)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.07.05.662870-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1)

<img width="850" alt="image" src="https://github.com/user-attachments/assets/f43be914-d6e7-4a71-8dda-146cc09a6c05" />

</div>

This repository contains the code for mRNABench, which benchmarks the embedding quality of genomic foundation models on mRNA specific tasks. The mRNABench contains a catalogue of datasets and training split logic which can be used to evaluate the embedding quality of several catalogued models.

**Paper:** [BioRxiv Link](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1)<br>
**Notebook Example:** [Colab Notebook](https://colab.research.google.com/drive/1VZF5NPwJYowAR3e6wuaiAuQyw2v7TSwx?usp=sharing)<br>
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

The inference-capable version of mRNABench can generate embeddings using all catalogued models.

```bash
conda create --name mrna_bench python=3.12
conda activate mrna_bench

pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
pip install -e .[base_models]
```

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
Evo2 is included in the base_models installation and works on GPUs with compute capability ≥ 7.0.

**HPC / multi-CUDA environments:** On systems with multiple CUDA versions installed (common on HPC clusters), TransformerEngine's NVRTC JIT compiler may pick up CUDA headers from a system-wide installation (e.g. `/usr/local/cuda`) rather than the conda environment's bundled CUDA headers, causing a compilation error like `NVRTC_ERROR_COMPILATION`. If you encounter this error, you can override it manually:
```bash
export NVTE_CUDA_INCLUDE_DIR="$CONDA_PREFIX/targets/x86_64-linux/include"
```

### Dev Mode
Dev mode allows generation of datasets from scratch and includes access to the RNA-Fazal localization dataset. See [Dev Mode Setup](#dev-mode-setup).


## Usage
Datasets can be retrieved using:

```python
import mrna_bench as mb

dataset = mb.load_dataset("go-mf")
data_df = dataset.data_df
```

The mRNABench can also be used to test out common genomic foundation models. The recommended way to load a model is via `mb.load_model()`, which automatically sets the model to inference mode (`.eval()` + gradients disabled):
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
embeddings = torch.stack(embeddings, dim=0).detach().cpu().numpy()

prober = (LinearProbeBuilder(dataset)
    .fetch_embedding_by_embedding_instance("orthrus-large-6", embeddings)
    .build()
)

metrics = prober.run_linear_probe(2541)
print(metrics)
```
`LinearProbeBuilder` now uses dataset metadata defaults for task, target,
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

Regression datasets keep the biological task name `regression`; choose the
estimator with `.set_regressor("ols")` or `.set_regressor("ridge")`. Results
are stored as `regression_ols` and `regression_ridge`. Legacy mRNABench
results stored under the `regression` name used RidgeCV.

The default homology split lazily extracts only the requested species from the
published homology map archive from the [Orthrus publication Zenodo](https://zenodo.org/records/13910050).
Custom thresholds query Ensembl Compara with the published grouping procedure;
the result is built once and cached by origin and threshold at
`homology_maps/ensembl-<version>/sim-<threshold>pct/<species>.csv`:

```python
from mrna_bench.data_splitter.homology_split import HomologySplitter

published = HomologySplitter(species="human")
ablated = HomologySplitter(
    species="human",
    similarity_threshold=50,
)
new_release = HomologySplitter(
    species="human",
    ensembl_version=115,
)
```

The published archive contains ten species. Their exact release-specific
paralog tables come from Ensembl Compara. Arbitrary species use their Ensembl
production name, for example `species="xenopus_tropicalis"`. The published maps
use Ensembl 110 and a strict 35% identity threshold. Raw paralog tables are
cached independently of threshold at
`homology_maps/ensembl-<version>/<species>.tsv`, so ablations reuse one download.


> [!CAUTION]
> Custom datasets must use gene names from the same Ensembl release as the
> Compara map (Ensembl 110 by default); unmatched names are treated as unrelated
> genes.

> [!NOTE]
> If you instantiate a model class directly (without `mb.load_model()`), call `model.set_inference_mode()` before embedding to ensure deterministic outputs. Call `model.set_train_mode()` when you need gradients, e.g. for fine-tuning.
> ```python
> model = RiNALMo("RiNALMo-giga", device)
> model.set_inference_mode()  # required for reproducible embeddings
> ```

Also see the `scripts/` folder for example scripts that uses slurm to embed dataset chunks in parallel for reduce runtime, as well as an example of multi-seed linear probing.

## Model Catalog
The models supported by the `base_models` installation are catalogued below.

### RNA Foundation Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **Orthrus** | `orthrus-base-4-track`<br>`orthrus-large-4-track`<br>`orthrus-large-6-track` | Mamba-based RNA foundation model pre-trained using contrastive learning on 45M RNA transcripts to capture functional and evolutionary relationships. 6-track version incorporates CDS and splice site information. | [[Code]](https://github.com/bowang-lab/Orthrus) [[Paper]](https://www.nature.com/articles/s41592-026-03064-3)|
| **RNA-FM** | `RNA-FM` | Transformer-based RNA foundation model pre-trained using MLM on 23M ncRNA sequences. | [[Github]](https://github.com/ml4bio/RNA-FM) |
| **mRNA-FM** | `mRNA-FM` | Transformer-based RNA foundation model pre-trained on mRNA CDS regions using a codon tokenizer. CDS track is required. | [[Github]](https://github.com/ml4bio/RNA-FM) |
| **SpliceBERT** | `SpliceBERT-1024nt`<br>`SpliceBERT-510nt`<br>`SpliceBERT-human-510nt` | Transformer-based RNA foundation model trained on 2M vertebrate mRNA sequences using MLM. Specialized for splice site prediction with human-only and context-length variants. | [[Github]](https://github.com/chenkenbio/SpliceBERT) |
| **RiNALMo** | `RiNALMo-micro`<br>`RiNALMo-mega`<br>`RiNALMo-giga` | Transformer-based RNA foundation model trained on 36M ncRNA sequences using MLM with modern architectural improvements including RoPE, SwiGLU activations, and Flash Attention. | [[Github]](https://github.com/lbcb-sci/RiNALMo) |
| **UTR-LM** | `UTR-LM-MLMSI`<br>`UTR-LM-MLMSISS`<br>`UTR-LM-MLM`<br>`UTR-LM-MLMSS` | Transformer-based RNA foundation model specialized for 5'UTR sequences. Pre-trained on random and endogenous UTR sequences from various species. | [[Github]](https://github.com/a96123155/UTR-LM) |
| **3UTRBERT** | `UTRBERT-3mer`<br>`UTRBERT-4mer`<br>`UTRBERT-5mer`<br>`UTRBERT-6mer` | Transformer-based RNA foundation model specialized for 3'UTR regions. Uses k-mer tokenization (3-6mers) and trained on 100k 3'UTR sequences. | [[Github]](https://github.com/yangyn533/3UTRBERT) |
| **RNA-MSM** | `RNA-MSM` | Structure-aware RNA foundation model trained using multiple sequence alignments from custom structure-based homology mapping across ~4000 RNA families. | [[Github]](https://github.com/yikunpku/RNA-MSM) |
| **RNAErnie** | `RNAErnie`<br>`RNAErnie2` | Transformer-based RNA foundation model trained using MLM with motif-level masking strategy on 23M ncRNA sequences. Uses contiguous token masking to learn RNA motifs. | [[Github]](https://github.com/CatIIIIIIII/RNAErnie) |
| **ERNIE-RNA** | `ERNIE-RNA`<br>`ERNIE-RNA-SS`<br>`ERNIE-RNA-MRL` | Transformer-based RNA foundation model with structural attention bias. Trained on 20M ncRNA sequences with custom attention incorporating RNA base pairing rules. SS version fine-tuned on structural tasks. | [[Github]](https://github.com/Bruce-ywj/ERNIE-RNA) |
| **RNABERT** | `RNABERT` | Transformer-based RNA foundation model with dual training objectives combining MLM and structural alignment learning. Trained on 80k ncRNA sequences. | [[Github]](https://github.com/mana438/RNABERT) |
| **CodonBERT** | `CodonBERT` | Transformer-based RNA foundation model trained on 10M+ mRNA sequences from mammals, bacteria, and viruses. Specialized for coding regions and mRNA properties. | [[Github]](https://github.com/Sanofi-Public/CodonBERT) |
| **Helix-mRNA** | `helix-mrna` | Hybrid Mamba2/Transformer model trained on 26M diverse eukaryotic and viral mRNAs. Features CDS-aware tokenization with special tokens at codon boundaries. | [[Github]](https://github.com/helicalAI/helical) |
| **mRNABERT** | `mRNABERT` | Transformer-based mRNA foundation model trained on 36M mRNA sequences using MLM with ALiBi positional embeddings and Flash Attention. Further pre-trained with contrastive learning to align CDS embeddings with protein embeddings from ProtT5-XL-UniRef50. | [[Github]](https://github.com/yyly6/mRNABERT) [[Paper]](https://www.nature.com/articles/s41467-025-65340-8) |
| **AIDO.RNA** | `AIDO.RNA-650M`<br>`AIDO.RNA-650M-CDS`<br>`AIDO.RNA-1.6B`<br>`AIDO.RNA-1.6B-CDS`<br>`AIDO.RNA-1M-MARS`<br>`AIDO.RNA-25M-MARS`<br>`AIDO.RNA-300M-MARS` | Transformer-based RNA foundation model trained using MLM on 42M ncRNA sequences. CDS-adapted variants are available for protein-coding sequences. Additional MARS pre-training variants are provided at 1M/25M/300M scale. | [[Github]](https://github.com/genbio-ai/ModelGenerator) [[Paper]](https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1) |
| **OmniGenome** | `omnigenome-52m`<br>`omnigenome-186m` | Transformer-based RNA foundation model pre-trained on plant RNA sequences from the OneKP initiative. Trained with three objectives: structure-contextualized masked token reconstruction (Str2Seq), sequence-to-structure prediction (Seq2Str), and MLM. Uses ViennaRNA-predicted secondary structures. | [[Github]](https://github.com/yangheng95/OmniGenBench) [[Paper]](https://arxiv.org/abs/2407.11242) |
| **Plant-RNAFM** | `plant_rnafm` | Transformer-based RNA foundation model pre-trained on 25M RNA sequences from 1,124 plant species (1KP). Trained with MLM, RNA secondary structure prediction (ViennaRNA), and RNA region annotation prediction (CDS, 5'UTR, 3'UTR). | [[Github]](https://github.com/yangheng95/PlantRNA-FM) [[Paper]](https://www.nature.com/articles/s42256-024-00946-z) |

### DNA Foundation Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **Borzoi** | `borzoi-replicate-0`<br>`borzoi-replicate-1`<br>`borzoi-replicate-2`<br>`borzoi-replicate-3`<br>`flashzoi-replicate-0`<br>`flashzoi-replicate-1`<br>`flashzoi-replicate-2`<br>`flashzoi-replicate-3`<br>`borzoi`<br>`flashzoi` | Deep learning model predicting RNA-seq coverage from DNA sequence. Hybrid architecture (convolutions + self-attention + U-Net) trained on 524 kb genomic windows. Flashzoi variants use Flash Attention for efficiency. | [[Github]](https://github.com/calico/borzoi) [[Paper]](https://www.nature.com/articles/s41588-024-02053-6) |
| **Enformer** | `enformer-official-rough` | Transformer-based model predicting functional genomic activity (RNA-seq, ATAC-seq, ChIP-seq) from 200 kb DNA windows. Combines convolutional stem with multi-head attention to capture long-range interactions. | [[Github]](https://github.com/lucidrains/enformer-pytorch) [[Paper]](https://www.nature.com/articles/s41592-021-01252-x) |
| **DNABERT** | `DNABERT-3mer`<br>`DNABERT-4mer`<br>`DNABERT-5mer`<br>`DNABERT-6mer` | Original BERT-based DNA foundation model using overlapping k-mer tokenization (3-6mers). Pre-trained using MLM on the human reference genome. | [[Github]](https://github.com/jerryji1993/DNABERT) |
| **DNABERT2** | `DNABERT2` | Modern Transformer-based DNA foundation model with BPE tokenization and rotary positional encoding. Pre-trained using MLM on multi-species genomic datasets. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_2) |
| **DNABERT-S** | `DNABERT-S` | Species-aware DNA foundation model trained with contrastive learning to encourage species grouping while discouraging cross-species associations. Covers microbial genomes including viruses, fungi, and bacteria. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_S) |
| **Carbon** | `Carbon-500M`<br>`Carbon-3B`<br>`Carbon-8B` | Decoder-only autoregressive genomic model with a Llama-style architecture (RoPE, `LlamaForCausalLM`) trained with next-token prediction on eukaryotic genes, mature/spliced mRNA, and bacterial genomes. Uses a hybrid 6-mer DNA + Qwen3 BPE tokenizer with native context of 8,192 tokens. | [[Github]](https://github.com/huggingface/carbon) |
| **Nucleotide Transformer** | `2.5b-multi-species`<br>`2.5b-1000g`<br>`500m-human-ref`<br>`500m-1000g`<br>`v2-50m-multi-species`<br>`v2-100m-multi-species`<br>`v2-250m-multi-species`<br>`v2-500m-multi-species` | Transformer-based DNA foundation model family with 6-mer tokenization. Available in multiple sizes (50M-2.5B parameters) trained on various genomic datasets from human reference to multi-species collections. | [[Github]](https://github.com/instadeepai/nucleotide-transformer) |
| **HyenaDNA** | `hyenadna-large-1m-seqlen-hf`<br>`hyenadna-medium-450k-seqlen-hf`<br>`hyenadna-medium-160k-seqlen-hf`<br>`hyenadna-small-32k-seqlen-hf`<br>`hyenadna-tiny-16k-seqlen-d128-hf` | Hyena-based DNA foundation model with near-linear scaling and ultra-long context capability. Pre-trained using next token prediction on human reference genome with various model sizes and sequence lengths. | [[Github]](https://github.com/HazyResearch/hyena-dna) |
| **Evo1** | `Evo1-1.5-7B-8K`<br>`Evo1-1-7B-8K`<br>`Evo1-1-7B-131K` | StripedHyena-based DNA foundation model trained autoregressively on OpenGenome dataset at single nucleotide, byte-level resolution. Offers near-linear scaling with ultra-long context variants up to 131k nucleotides. | [[Github]](https://github.com/evo-design/evo) |
| **Evo2** | `Evo2-1B-8K`<br>`Evo2-7B-8K`<br>`Evo2-7B-262K`<br>`Evo2-7B-1M`<br>`Evo2-20B-1M`<br>`Evo2-40B-8K`<br>`Evo2-40B-1M` | Next-generation StripedHyena2-based DNA foundation model trained on OpenGenome2 dataset. Provides multi-layer embeddings with ultra-long context capability up to 1M nucleotides for large model variants. Loaded via a standalone HuggingFace port (no `evo2` pip package required). | [[Github]](https://github.com/ArcInstitute/evo2) |
| **AlphaGenome** | `alphagenome` | Deep learning model predicting functional genomic activity from DNA sequence. Uses a transformer architecture built on convolutional layers and self-attention to model long-range interactions, trained on 200 kb genomic windows across human and mouse datasets. Uses the GenomicsXAI PyTorch implementation of DeepMind's AlphaGenome. | [[Github]](https://github.com/google-deepmind/alphagenome_research) [[Port]](https://github.com/genomicsxai/alphagenome-pytorch) |
| **NucleotideTransformerV3** | `v3_8M_pre`<br>`v3_100M_pre`<br>`v3_650M_pre`<br>`v3_100M_post`<br>`v3_650M_post` | Transformer/CNN U-Net DNA foundation model trained on OpenGenome2 with MLM at single-nucleotide resolution. Downsamples with convolutions, processes with a Transformer bottleneck, and upsamples — enabling ultra-long sequences up to 1 MB. Post-trained variants were further trained to predict 16K+ genomic tracks. | [[Github]](https://github.com/instadeepai/nucleotide-transformer) [[Paper]](https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1) |
| **GENERanno** | `prokaryote-0.5b-base`<br>`prokaryote-0.5b-cds-annotator`<br>`eukaryote-0.5b-base`<br>`eukaryote-1.2b-cds-annotator-preview` | Transformer-encoder genomic foundation model for metagenomic annotation at single-nucleotide resolution with bidirectional attention over sequences up to 8k bp. Pre-trained on 715B bp prokaryotic and 386B bp eukaryotic sequences. CDS-annotator variants are fine-tuned for metagenomic CDS calling. | [[Github]](https://github.com/GenerTeam/GENERanno) [[Paper]](https://www.biorxiv.org/content/10.1101/2025.06.04.656517v3) |
| **GENERator** | `eukaryote-1.2b-base`<br>`v2-eukaryote-1.2b-base`<br>`v2-prokaryote-1.2b-base`<br>`eukaryote-3b-base`<br>`v2-eukaryote-3b-base`<br>`v2-prokaryote-3b-base` | Autoregressive Transformer genomic foundation model using k-mer tokenization trained on gene-centric functional regions. V2 introduces Factorized Nucleotide Supervision (FNS) and Genome Compression Pretraining (GCP), with context up to 98k bp and eukaryotic/prokaryotic variants. | [[Github]](https://github.com/GenerTeam/GENERator) [[Paper]](https://arxiv.org/abs/2502.07272) |


### Baseline Models

| Model Name | Model Versions | Description | Citation |
| :--------: | :------------- | ----------- | :------: |
| **NaiveBaseline** | `naive-4-track` | Non-neural baseline using traditional sequence features including k-mer counts (3-7mers), GC content, and sequence statistics. | N/A |
| **NaiveBaselineSixTrack** | `naive-6-track` | Six-track variant of NaiveBaseline that additionally incorporates CDS and splice-site information (e.g. CDS length and exon count) from structural annotations. | N/A |
| **NaiveMamba** | `naive-mamba` | Randomly initialized Mamba model serving as an untrained baseline. Uses 6-track input (sequence + CDS + splice information) with fixed random seed for reproducible comparisons. | N/A |

### Adding a new model
All models should inherit from `EmbeddingModel`. Each model file should lazily load dependencies within `__init__` so models remain independently installable. Models implement `embed()` and declare concrete `ModelBehavior` values; likelihood models expose `logits()` and `sequence_score()` with optional aligned `cds` and `splice` tracks, while sequence-to-function models can expose `predict_tracks()`. Pseudo-likelihood models also expose `masked_marginal_llr()`, which masks every tokenizer position changed by a substitution, including overlapping k-mer or codon tokens. FlashAttention wrappers use FP16 by default (with model-specific overrides where FP16 is unstable), and default mean pooling is performed in FP32. New models should be added to `MODEL_CATALOG`.

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
| RNA Lifecycle (Ietswaart) | <code>rna&#8209;lifecycle&#8209;ietswaart</code> | Lower-tertile normalized coverage share across chromatin, cytoplasm, and polysome fractions in human K562 cells. | `multilabel` | [paper](https://pubmed.ncbi.nlm.nih.gov/38964322/) |

The transcript tables used to rebuild this dataset are stored with Git LFS at
`resources/ietswaart_wf_transcript_tables.tar.gz` and excluded from the Python
wheel. Installed copies download them only when
`RNALifecycleIetswaart(force_rebuild_raw=True)` is requested. Processing is
documented in `resources/README.md`.

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

#### Embedding-based VEP

Variant effects can be scored in several ways, including likelihood ratios,
masked-marginal scores, and differences between reference and alternate
embeddings. The default mRNABench embedding score is the L2 norm of the pooled
alternate minus reference embedding.

Embedding differences can be sensitive to small floating-point changes because
they subtract two nearly identical vectors. In our checks, AIDO.DNA, DNABERT2,
GENERanno, RiNALMo, mRNABERT, and Evo1 showed meaningful score changes between
attention backends. This does not mean that one backend is always better, but
it does mean that attention backend, dtype, and pooling are part of the VEP
method. Before running embedding-based VEP at scale, compare the available
backends on a representative subset and check that variant rankings and
metrics are stable. Reference and alternate embeddings should always be
generated with the same settings.

Likelihood-based VEP is available through
`scripts/linear_probe/by_modelname.py --task likelihood_vep` with
`--score_method causal_likelihood`, `pseudo_likelihood`, or
`masked_marginal`. Row-wise VEP datasets are paired with their wild-type
transcripts automatically, and results are stored in `results.db` under the
`likelihood_vep` task with a method-and-normalization-specific key. The
database stores aggregate metrics, not raw logits or per-variant scores.

PLLR masks every non-special model token and can therefore be expensive on
long sequences; `--score_batch_size` controls how many masked inputs are
evaluated together. Long sequences are supported through chunking, but chunks
are scored independently without cross-chunk context. Masked-marginal scoring
is intended for substitutions and excludes indels with a warning. Causal and
pseudo likelihood ratios default to summed log probabilities; use
`--normalization mean` only when a length-normalized score is desired.

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

The original sources for each dataset and model should be cited if used, and can be found above.

## NucleotideTransformerV3 Setup
Please note that as of now, NucleotideTransformerV3 is a gated model and requires applying for access through HuggingFace. If you get access, you need to create a HuggingFace token at https://huggingface.co/settings/tokens and and then run `hf auth login` to enter your token for access.

## Helix-mRNA Setup
Please note that as of now, Helix-mRNA is a gated model and requires applying for access through HuggingFace. If you get access, you need to create a HuggingFace token at https://huggingface.co/settings/tokens and and then run `hf auth login` to enter your token for access.

## Dev Mode Setup
Dev mode requires additional dependencies for generating datasets from scratch and accessing certain datasets.

```bash
conda create --name mrna_bench_dev python=3.12
conda activate mrna_bench_dev

pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
pip install -e .[base_models,dev]
```
