from .aido import AIDORNA
from .dnabert import DNABERT2
from .nucleotide_transformer import NucleotideTransformer
from .orthrus import Orthrus
from .rnafm import RNAFM

from .embedding_model import EmbeddingModel


MODEL_CATALOG: dict[str, EmbeddingModel] = {
    "AIDO.RNA": AIDORNA,
    "DNABERT2": DNABERT2,
    "NucleotideTransformer": NucleotideTransformer,
    "Orthrus": Orthrus,
    "RNA-FM": RNAFM,
}
