from .aido import AIDORNA
from .orthrus import Orthrus
from .embedding_model import EmbeddingModel


MODEL_CATALOG: dict[str, EmbeddingModel] = {
    "AIDO.RNA": AIDORNA,
    "Orthrus": Orthrus
}
