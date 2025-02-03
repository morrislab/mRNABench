from typing import Type

from .aido import AIDORNA
from .dnabert import DNABERT2
from .ernierna import ERNIERNA
from .hyenadna import HyenaDNA
from .nucleotide_transformer import NucleotideTransformer
from .orthrus import Orthrus
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
    "DNABERT2": DNABERT2,
    "ERNIE-RNA": ERNIERNA,
    "HyenaDNA": HyenaDNA,
    "NucleotideTransformer": NucleotideTransformer,
    "RiNALMo": RiNALMo,
    "Orthrus": Orthrus,
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
    "DNABERT2": ["dnabert2"],
    "ERNIE-RNA": ["ernierna", "ernierna-ss"],
    "HyenaDNA": [
        "hyenadna-large-1m-seqlen-hf",
        "hyenadna-medium-450k-seqlen-hf",
        "hyenadna-medium-160k-seqlen-hf",
        "hyenadna-small-32k-seqlen-hf",
        "hyenadna-tiny-16k-seqlen-d128-hf"
    ],
    "NucleotideTransformer": [
        "2.5b-multi-species",
        "2.5b-1000g",
        "500m-human-ref",
        "500m-1000g",
        "v2-50m-multi-species",
        "v2-100m-multi-species",
        "v2-250m-multi-species",
        "v2-500m-multi-species"
    ],
    "Orthrus": [
        "orthrus-large-6-track",
        "orthrus-base-4-track"
    ],
    "RiNALMo": ["rinalmo"],
    "RNABERT": ["rnabert"],
    "RNAErnie": ["rnaernie"],
    "RNA-FM": ["rna-fm", "mrna-fm"],
    "RNA-MSM": ["rnamsm"],
    "SpliceBERT": [
        "SpliceBERT.1024nt",
        "SpliceBERT-human.510nt",
        "SpliceBERT.510nt"
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
