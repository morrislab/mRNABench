from typing import Type

from .aido import AIDORNA
from .borzoi import Borzoi
from .codonbert import CodonBERT
from .dnabert import DNABERT2
from .dnabert_s import DNABERTS
from .enformer import Enformer
from .ernierna import ERNIERNA
from .evo1 import Evo1
from .evo2 import Evo2
from .generanno import GENERanno
from .generator import GENERator
from .helix_mrna import HelixmRNAWrapper
from .hyenadna import HyenaDNA
from .mrnabert import mRNABERT
from .naive_baseline import NaiveBaseline
from .naive_mamba import NaiveMamba
from .nucleotide_transformer import NucleotideTransformer
from .nucleotide_transformer_v3 import NucleotideTransformerV3
from .omnigenome import OmniGenome
from .orthrus import Orthrus
from .plant_rnafm import PlantRNAFM
from .rinalmo import RiNALMo
from .rnabert import RNABERT
from .rnaernie import RNAErnie
from .rnafm import RNAFM
from .rnamsm import RNAMSM
from .splicebert import SpliceBERT
from .utrbert import UTRBERT
from .utrlm import UTRLM

from .embedding_model import EmbeddingModel


MODEL_CATALOG: dict[str, Type[EmbeddingModel]] = {
    "AIDO.RNA": AIDORNA,
    "Borzoi": Borzoi,
    "CodonBERT": CodonBERT,
    "DNABERT-S": DNABERTS,
    "DNABERT2": DNABERT2,
    "Enformer": Enformer,
    "ERNIE-RNA": ERNIERNA,
    "Evo1": Evo1,
    "Evo2": Evo2,
    "GENERanno": GENERanno,
    "GENERator": GENERator,
    "Helix-mRNA": HelixmRNAWrapper,
    "HyenaDNA": HyenaDNA,
    "mRNABERT": mRNABERT,
    "NaiveBaseline": NaiveBaseline,
    "NaiveMamba": NaiveMamba,
    "NucleotideTransformer": NucleotideTransformer,
    "NucleotideTransformerV3": NucleotideTransformerV3,
    "RiNALMo": RiNALMo,
    "OmniGenome": OmniGenome,
    "Orthrus": Orthrus,
    "Plant-RNAFM": PlantRNAFM,
    "RNABERT": RNABERT,
    "RNAErnie": RNAErnie,
    "RNA-FM": RNAFM,
    "RNA-MSM": RNAMSM,
    "SpliceBERT": SpliceBERT,
    "3UTRBERT": UTRBERT,
    "UTR-LM": UTRLM,
}


MODEL_VERSION_MAP: dict[str, list[str]] = {
    "AIDO.RNA": [
        "aido_rna_650m",
        "aido_rna_650m_cds",
        "aido_rna_1b600m",
        "aido_rna_1b600m_cds"
    ],
    "Borzoi": [
        "borzoi-replicate-0",
        "borzoi-replicate-1",
        "borzoi-replicate-2",
        "borzoi-replicate-3",
        "flashzoi-replicate-0",
        "flashzoi-replicate-1",
        "flashzoi-replicate-2",
        "flashzoi-replicate-3",
        "borzoi",
        "flashzoi"
    ],
    "CodonBERT": ["codonbert"],
    "DNABERT-S": ["dnabert-s"],
    "DNABERT2": ["dnabert2"],
    "Enformer": ["enformer-official-rough"],
    "ERNIE-RNA": ["ernierna", "ernierna-ss"],
    "Evo1": [
        "evo-1.5-8k-base",
        "evo-1-8k-base",
        "evo-1-131k-base"
    ],
    "Evo2": [
        "evo2_1b_base",
        "evo2_7b_base",
        "evo2_7b",
        "evo2_7b_262k",
        "evo2_20b",
        "evo2_40b_base",
        "evo2_40b"
    ],
    "GENERanno": [
        "prokaryote-0.5b-base",
        "prokaryote-0.5b-cds-annotator",
        "eukaryote-0.5b-base",
        "eukaryote-1.2b-cds-annotator-preview"
    ],
    "GENERator": [
        "eukaryote-1.2b-base",
        "v2-eukaryote-1.2b-base",
        "v2-prokaryote-1.2b-base",
        "eukaryote-3b-base",
        "v2-eukaryote-3b-base",
        "v2-prokaryote-3b-base"
    ],
    "Helix-mRNA": ["helix-mrna"],
    "HyenaDNA": [
        "hyenadna-tiny-16k-seqlen-d128-hf",
        "hyenadna-small-32k-seqlen-hf",
        "hyenadna-medium-160k-seqlen-hf",
        "hyenadna-medium-450k-seqlen-hf",
        "hyenadna-large-1m-seqlen-hf"
    ],
    "mRNABERT": ["mRNABERT"],
    "NaiveBaseline": [
        "naive-4-track",
        "naive-6-track"
    ],
    "NaiveMamba": [
        "naive-mamba"
    ],
    "NucleotideTransformer": [
        "v2-50m-multi-species",
        "v2-100m-multi-species",
        "v2-250m-multi-species",
        "v2-500m-multi-species",
        "500m-human-ref",
        "500m-1000g",
        "2.5b-multi-species",
        "2.5b-1000g"
    ],
    "NucleotideTransformerV3": [
        "v3_8M_pre",
        "v3_100M_pre",
        "v3_100M_post",
        "v3_650M_pre",
        "v3_650M_post"
    ],
    "OmniGenome": [
        "omnigenome-52m",
        "omnigenome-186m"
    ],
    "Orthrus": [
        "orthrus-base-4-track",
        "orthrus-large-6-track"
    ],
    "Plant-RNAFM": ["plant_rnafm"],
    "RiNALMo": [
        "rinalmo-micro",
        "rinalmo-mega",
        "rinalmo-giga"
    ],
    "RNABERT": ["rnabert"],
    "RNAErnie": ["rnaernie"],
    "RNA-FM": ["rna-fm", "mrna-fm"],
    "RNA-MSM": ["rnamsm"],
    "SpliceBERT": [
        "SpliceBERT-human.510nt",
        "SpliceBERT.510nt",
        "SpliceBERT.1024nt",
    ],
    "3UTRBERT": [
        "utrbert-3mer",
        "utrbert-4mer",
        "utrbert-5mer",
        "utrbert-6mer",
        "utrbert-3mer-utronly",
        "utrbert-4mer-utronly",
        "utrbert-5mer-utronly",
        "utrbert-6mer-utronly"
    ],
    "UTR-LM": [
        "utrlm-te_el",
        "utrlm-mrl",
        "utrlm-te_el-utronly",
        "utrlm-mrl-utronly"
    ]
}
