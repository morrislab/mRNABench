from typing import Type

from .aido import AIDORNA
from .dnabert import DNABERT2
from .hyenadna import HyenaDNA
from .nucleotide_transformer import NucleotideTransformer
from .orthrus import Orthrus
from .rnafm import RNAFM

from .embedding_model import EmbeddingModel


MODEL_CATALOG: dict[str, Type[EmbeddingModel]] = {
    "AIDO.RNA": AIDORNA,
    "DNABERT2": DNABERT2,
    "NucleotideTransformer": NucleotideTransformer,
    "Orthrus": Orthrus,
    "RNA-FM": RNAFM,
    "HyenaDNA": HyenaDNA
}
