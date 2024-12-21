from .go_mol_func import GOMolecularFunction
from .pcg_essentiality import PCGEssentiality
from .lncrna_essentiality import LNCRNAEssentiality


DATASET_CATALOG = {
    "go-mf": GOMolecularFunction,
    "pcg-ess": PCGEssentiality,
    "lncrna-ess": LNCRNAEssentiality,
}
