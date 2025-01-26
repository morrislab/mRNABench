from typing import Type

from .aido import AIDORNA
from .dnabert import DNABERT2
from .hyenadna import HyenaDNA
from .nucleotide_transformer import NucleotideTransformer
from .orthrus import Orthrus
from .rnafm import RNAFM
from .splicebert import SpliceBERT

from .embedding_model import EmbeddingModel


MODEL_CATALOG: dict[str, Type[EmbeddingModel]] = {
    "AIDO.RNA": AIDORNA,
    "DNABERT2": DNABERT2,
    "HyenaDNA": HyenaDNA,
    "NucleotideTransformer": NucleotideTransformer,
    "Orthrus": Orthrus,
    "RNA-FM": RNAFM,
    "SpliceBERT": SpliceBERT
}


MODEL_VERSION_MAP: dict[str, list[str]] = {
    "AIDO.RNA": [
        "aido_rna_650m",
        "aido_rna_650m_cds",
        "aido_rna_1b600m",
        "aido_rna_1b600m_cds"
    ],
    "DNABERT2": ["dnabert2"],
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
    "RNA-FM": [
        "rna-fm",
        "mrna-fm"
    ],
    "SpliceBERT": [
        "SpliceBERT.1024nt",
        "SpliceBERT-human.510nt",
        "SpliceBERT.510nt"
    ]
}
