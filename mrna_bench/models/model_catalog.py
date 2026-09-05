from typing import Type

from .aido_dna import AIDODNA
from .aido_rna import AIDORNA
from .alphagenome import AlphaGenome
from .borzoi import Borzoi
from .carbon import Carbon
from .codonbert import CodonBERT
from .dnabert2 import DNABERT2
from .dnabert import DNABERT
from .dnabert_s import DNABERTS
from .enformer import Enformer
from .ernierna import ERNIERNA
from .genalm import GenaLM
from .glm2 import GLM2
from .evo1 import Evo1
from .evo2 import Evo2
from .generanno import GENERanno
from .generator import GENERator
from .helix_mrna import HelixmRNA
from .hyenadna import HyenaDNA
from .moderngena import ModernGENA
from .mrnabert import mRNABERT
from .naive_baseline import NaiveBaseline, NaiveBaselineSixTrack
from .naive_mamba import NaiveMamba
from .nucleotide_transformer import NucleotideTransformer
from .nucleotide_transformer_v3 import NucleotideTransformerV3
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
    "AIDO.DNA": AIDODNA,
    "AIDO.RNA": AIDORNA,
    "AlphaGenome": AlphaGenome,
    "Borzoi": Borzoi,
    "Carbon": Carbon,
    "CodonBERT": CodonBERT,
    "DNABERT": DNABERT,
    "DNABERT-S": DNABERTS,
    "DNABERT2": DNABERT2,
    "Enformer": Enformer,
    "ERNIE-RNA": ERNIERNA,
    "Evo1": Evo1,
    "GENA-LM": GenaLM,
    "gLM2": GLM2,
    "Evo2": Evo2,
    "GENERanno": GENERanno,
    "GENERator": GENERator,
    "Helix-mRNA": HelixmRNA,
    "HyenaDNA": HyenaDNA,
    "ModernGENA": ModernGENA,
    "mRNABERT": mRNABERT,
    "mRNA-FM": MRNAFM,
    "NaiveBaseline": NaiveBaseline,
    "NaiveBaselineSixTrack": NaiveBaselineSixTrack,
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
    name: model_class.valid_versions
    for name, model_class in MODEL_CATALOG.items()
}
