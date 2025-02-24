from collections.abc import Callable

from .benchmark_dataset import BenchmarkDataset
from .go_mol_func import GOMolecularFunction
from .pcg_essentiality import (
    PCGEssHAP1,
    PCGEssHEK293FT,
    PCGEssK562,
    PCGEssMDA_MB_231,
    PCGEssTHP1,
    PCGEssShared
)
from .lncrna_essentiality import (
    LNCRNAEssHAP1,
    LNCRNAEssHEK293FT,
    LNCRNAEssK562,
    LNCRNAEssMDA_MB_231,
    LNCRNAEssTHP1,
    LNCRNAEssShared
)
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .prot_loc import ProteinLocalization
from .prot_loc_van_nostrand import ProteinLocalizationVan
from .mrl_sugimoto import MRLSugimoto
from .mrl_sample import (
    MRLSampleEGFP,
    MRLSampleMCherry,
    MRLSampleDesigned,
    MRLSampleVarying
)

DATASET_CATALOG: dict[str, Callable[..., BenchmarkDataset]] = {
    "go-mf": GOMolecularFunction,
    "pcg-ess-hap1": PCGEssHAP1,
    "pcg-ess-hek293ft": PCGEssHEK293FT,
    "pcg-ess-k562": PCGEssK562,
    "pcg-ess-mda-mb-231": PCGEssMDA_MB_231,
    "pcg-ess-thp1": PCGEssTHP1,
    "pcg-ess-shared": PCGEssShared,
    "lncrna-ess-hap1": LNCRNAEssHAP1,
    "lncrna-ess-hek293ft": LNCRNAEssHEK293FT,
    "lncrna-ess-k562": LNCRNAEssK562,
    "lncrna-ess-mda-mb-231": LNCRNAEssMDA_MB_231,
    "lncrna-ess-thp1": LNCRNAEssTHP1,
    "lncrna-ess-shared": LNCRNAEssShared,
    "rnahl-human": RNAHalfLifeHuman,
    "rnahl-mouse": RNAHalfLifeMouse,
    "prot-loc": ProteinLocalization,
    "prot-loc-van": ProteinLocalizationVan,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
}

DATASET_DEFAULT_TASK: dict[str, str] = {
    "go-mf": "multilabel",
    "rnahl-human": "reg_ridge",
    "rnahl-mouse": "reg_ridge",
    "prot-loc": "multilabel",
    "mrl-sugimoto": "reg_ridge",
    # "pcg-ess-hap1": "classification",
    # "pcg-ess-hek293ft": "classification",
    # "pcg-ess-k562": "classification",
    # "pcg-ess-mda-mb-231": "classification",
    # "pcg-ess-thp1": "classification",
    # "pcg-ess-shared": "classification",
    # "lncrna-ess-hap1": "classification",
    # "lncrna-ess-hek293ft": "classification",
    # "lncrna-ess-k562": "classification",
    # "lncrna-ess-mda-mb-231": "classification",
    # "lncrna-ess-thp1": "classification",
    # "lncrna-ess-shared": "classification",
}
