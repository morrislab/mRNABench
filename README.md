# mRNABench
This repository contains a workflow to benchmark the embedding quality of genomic foundation models on mRNA specific tasks. The mRNABench contains a catalogue of datasets and training split logic which can be used to evaluate the embedding quality of several catalogued models.

**Jump to:** [Model Catalog](#model-catalog) [Dataset Catalog](#dataset-catalog)

## Setup
Several configurations of the mRNABench are available.

### Datasets Only
If you are interested in the benchmark datasets **only**, you can run:

```bash
pip install mrna-bench
```

### Base Models
The inference-capable version of mRNABench that can generate embeddings using
most models (except Evo2 and Helix mRNA) can be installed as shown below. Note that this requires PyTorch version
2.2.2 with CUDA 12.1.

```bash
conda create --name mrna_bench python=3.10
conda activate mrna_bench

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install mrna-bench[base_models]
```
Inference with other models will require the installation of the model's
dependencies first, which are usually listed on the model's GitHub page (see below).

### Evo2
Inference using Evo2 requires installing the following in its own
environment. Note, I had an issue where the evo_40b models, when downloaded,
had their merged checkpoints stored one directory above the huggingface hub.
I had to manually move the checkpoint into its corresponding snapshot directory.
/hub/models--arcinstitute-evo2_40b*/snapshots/snapshot_name/

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

### Post-install
> [!IMPORTANT]
> After installation, please run the following in Python to set where data associated with the benchmarks will be stored.
```python
import mrna_bench as mb

path_to_dir_to_store_data = "DESIRED_PATH"
mb.update_data_path(path_to_dir_to_store_data)

path_to_dir_to_store_weights = "/data1/morrisq/ian/rna_benchmarks/model_weights"
mb.update_model_weights_path(path_to_dir_to_store_weights)
```

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

| Model Name | &nbsp;&nbsp;Model&nbsp;Versions&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description | Citation |
| :--------: |  ---------------------- | ------------- | :------: |
| `Orthrus` | `orthrus-large-6-track`<br> `orthrus-base-4-track` | Mamba-based RNA FM pre-trained using contrastive learning on ~45M RNA transcripts to capture functional and evolutionary relationships. | [[Code]](https://github.com/bowang-lab/Orthrus) [[Paper]](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v2)|
| `RNA-FM` | `rna-fm` <br> `mrna-fm` | Transformer-based RNA FM pre-trained using MLM on 23M ncRNA sequences. mRNA-FM trained on mRNA CDS regions using codon tokenizer. | [[Github]](https://github.com/ml4bio/RNA-FM) |
| `DNABERT2` | `dnabert2` | Transformer-based DNA FM pre-trained using MLM on multispecies genomic dataset. Uses BPE and other modern architectural improvements for efficiency. | [[Github]](https://github.com/MAGICS-LAB/DNABERT_2) |
| `Nucleotide`<br>`Transformer` | `2.5b-multi-species` <br> `2.5b-1000g` <br> `500m-human-ref` <br> `500m-1000g` <br> `v2-50m-multi-species` <br> `v2-100m-multi-species` <br> `v2-250m-multi-species` <br> `v2-500m-multi-species` | Transformer-based DNA FM pre-trained using MLM on a variety of possible datasets at various model sizes. Sequence is tokenized using 6-mers. | [[Github]](https://github.com/instadeepai/nucleotide-transformer) |
| `HyenaDNA` | `hyenadna-large-1m-seqlen-hf` <br> `hyenadna-medium-450k-seqlen-hf` <br> `hyenadna-medium-160k-seqlen-hf` <br> `hyenadna-small-32k-seqlen-hf` <br> `hyenadna-tiny-16k-seqlen-d128-hf` | Hyena-based DNA FM pre-trained using NTP on the human reference genome. Available at various model sizes and pretraining sequence contexts. | [[Github]](https://github.com/HazyResearch/hyena-dna) |
| `SpliceBERT` | `SpliceBERT.1024nt` <br> `SpliceBERT-human.510nt` <br> `SpliceBERT.510nt` |  Transformer-based RNA foundation model trained on 2M vertebrate mRNA sequences using MLM. Alternative versions trained on only human RNA, and with smaller context windows. | [[Github]](https://github.com/chenkenbio/SpliceBERT) |
| `RiNALMo` | `rinalmo` | Transformer-based RNA foundation model trained on 36M ncRNA sequences using MLM and other modern architectural improvements such as RoPE, SwiGLU activations, and Flash Attention. | [[Github]](https://github.com/lbcb-sci/RiNALMo) |
| `UTR-LM` | `utrlm-te_el` <br> `utrlm-mrl` | Transformer-based RNA foundation model that is pre-trained on random and endogenous 5'UTR sequences from various species using MLM. | [[Github]](https://github.com/a96123155/UTR-LM) |
| `3UTRBERT` | `utrbert-3mer` <br> `utrbert-4mer` <br> `utrbert-5mer` <br> `utrbert-6mer` | Transformer-based RNA foundation model that is pre-trained on the 3'UTR regions of 100K RNA sequences using MLM. | [[Github]](https://github.com/yangyn533/3UTRBERT) |
| `RNA-MSM` | `rnamsm` | Transformer-based RNA foundation model trained by using MSA from custom structure-based homology map on roughly 8M RNA sequences. | [[Github]](https://github.com/yikunpku/RNA-MSM) |
| `RNAErnie` | `rnaernie` | Transformer-based RNA foundation model trained using MLM at various mask sizes on 23M ncRNA sequences. | [[Github]](https://github.com/CatIIIIIIII/RNAErnie) |
| `RNABERT` | `rnabert` | Transformer-based RNA foundation model trained using MLM and a structural alignment objective on 80K ncRNA sequences | [[Github]](https://github.com/mana438/RNABERT) |
| `ERNIE-RNA` | `ernierna` <br> `ernierna-ss` | Transformer-based RNA foundation model trained using MLM with structural information added as attention mask biases. Pretrained on 20M ncRNA sequences. | [[Github]](https://github.com/Bruce-ywj/ERNIE-RNA) |


Many of the models wrappers (3UTRBERT, RiNALMo, UTR-LM, RNA-MSM, RNAErnie) use reimplementations from the `multimolecule` package. See their [website](https://multimolecule.danling.org/) for more details.

### Adding a new model
All models should inherit from the template `EmbeddingModel`. Each model file should lazily load dependencies within its `__init__` methods so each model can be used individually without install all other models. Models must implement `get_model_short_name(model_version)` which fetches the internal name for the model. This must be unique for every model version and must not contain underscores. Models should implement either `embed_sequence` or `embed_sequence_sixtrack` (see code for method signature). New models should be added to `MODEL_CATALOG`.

## Dataset Catalog
The current datasets catalogued are:
| Dataset Name | Catalogue Identifier | Description | Tasks | Citation |
|---|---|---|---|---|
| GO Molecular Function | <code>go-mf</code> | Classification of the molecular function of a transcript's  product as defined by the GO Resource. | `multilabel` | [website](https://geneontology.org/) |
| Mean Ribosome Load (Sugimoto) | <code>mrl&#8209;sugimoto</code> | Mean ribosome load (MRL) per transcript isoform as measured in Sugimoto et al. 2022. | `regression` | [paper](https://www.nature.com/articles/s41594-022-00819-2) |
| RNA Half-life (Human) | <code>rnahl&#8209;human</code> | RNA half-life of human transcripts collected by Agarwal et al. 2022. | `regression` | [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) |
| RNA Half-life (Mouse) | <code>rnahl&#8209;mouse</code> | RNA half-life of mouse transcripts collected by Agarwal et al. 2022. | `regression` | [paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) |
| Protein Subcellular Localization | <code>prot&#8209;loc</code> | Subcellular localization of transcript protein product defined in Protein Atlas. | `multilabel` | [website](https://www.proteinatlas.org/) |
| Mean Ribosome Load (Sample) | <code>mrl&#8209;sample&#8209;egfp</code> <br><code>mrl&#8209;sample&#8209;mcherry</code><br><code>mrl&#8209;sample&#8209;designed</code><br><code>mrl&#8209;sample&#8209;varying</code> | Mean ribosome load (MRL) measured in an MPRA of both random and designed 5'UTR regions (50nts) attached to a construct with either eGFP or mCherry. | `regression` | [paper](https://pubmed.ncbi.nlm.nih.gov/31267113/)|
| Protein Coding Gene Essentiality | <code>pcg&#8209;ess</code> | Essentiality of PCGs as measured by CRISPR knockdown. Log-fold expression and binary essentiality available on several cell lines. | `regression` `classification`| [paper](https://www.cell.com/cell/fulltext/S0092-8674(24)01203-0)|

### Adding a new dataset
New datasets should inherit from `BenchmarkDataset`. Dataset names cannot contain underscores. Each new dataset should download raw data and process it into a dataframe by overriding `process_raw_data`. This dataframe should store transcript as rows, using string encoding in the `sequence` column. If homology splitting is required, a column `gene` containing gene names is required. Six track embedding also requires columns `cds` and `splice`. The target column can have any name, as it is specified at time of probing. New datasets should be added to `DATASET_CATALOG`.
