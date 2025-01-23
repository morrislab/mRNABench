from collections.abc import Callable

from .benchmark_dataset import BenchmarkDataset
from .go_mol_func import GOMolecularFunction
from .pcg_essentiality import PCGEssentiality
from .lncrna_essentiality import LNCRNAEssentiality
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .prot_loc import ProteinLocalization
from .mrl_sugimoto import MRLSugimoto

DATASET_CATALOG: dict[str, Callable[..., BenchmarkDataset]] = {
    "go-mf": GOMolecularFunction,
    "pcg-ess": PCGEssentiality,
    "lncrna-ess": LNCRNAEssentiality,
    "rnahl-human": RNAHalfLifeHuman,
    "rnahl-mouse": RNAHalfLifeMouse,
    "prot-loc": ProteinLocalization,
    "mrl-sugimoto": MRLSugimoto,
}
