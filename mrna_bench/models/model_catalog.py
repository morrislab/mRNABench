from typing import Type

from .borzoi import Borzoi
from .codonbert import CodonBERT
from .dnabert import DNABERT2
from .dnabert_s import DNABERTS
from .enformer import Enformer
from .ernierna import ERNIERNA
from .evo1 import Evo1
from .evo2 import Evo2
from .helix_mrna import HelixmRNAWrapper
from .hyenadna import HyenaDNA
from .mrnabert import mRNABERT
from .naive_baseline import NaiveBaseline, NaiveBaselineSixTrack
from .naive_mamba import NaiveMamba
from .nucleotide_transformer import NucleotideTransformer
from .omnigenome import OmniGenome
from .orthrus import Orthrus
from .mrnafm import MRNAFM
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
    "Borzoi": Borzoi,
    "CodonBERT": CodonBERT,
    "DNABERT-S": DNABERTS,
    "DNABERT2": DNABERT2,
    "Enformer": Enformer,
    "ERNIE-RNA": ERNIERNA,
    "Evo1": Evo1,
    "Evo2": Evo2,
    "Helix-mRNA": HelixmRNAWrapper,
    "HyenaDNA": HyenaDNA,
    "mRNABERT": mRNABERT,
    "mRNA-FM": MRNAFM,
    "NaiveBaseline": NaiveBaseline,
    "NaiveBaselineSixTrack": NaiveBaselineSixTrack,
    "NaiveMamba": NaiveMamba,
    "NucleotideTransformer": NucleotideTransformer,
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
    name: model_class.valid_versions
    for name, model_class in MODEL_CATALOG.items()
}
