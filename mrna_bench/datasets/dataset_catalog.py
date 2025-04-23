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
from .rna_loc_ietswaart import RNALocalizationIetswaart
from .prot_loc import ProteinLocalization
from .mrl_sugimoto import MRLSugimoto
from .mrl_sample import (
    MRLSampleEGFP,
    MRLSampleMCherry,
    MRLSampleDesigned,
    MRLSampleVarying
)

from .eclip_binding import (
    eCLIPBindingK562,
    eCLIP_K562_RBPS_LIST,
    eCLIPBindingHepG2,
    eCLIP_HepG2_RBPS_LIST,
)

DATASET_CATALOG: dict[str, Callable[..., BenchmarkDataset]] = {
    "eclip-binding-k562": eCLIPBindingK562,
    "eclip-binding-hepg2": eCLIPBindingHepG2,
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
    "rna-loc-ietswaart": RNALocalizationIetswaart,
    "prot-loc": ProteinLocalization,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
}

DATASET_INFO = {
    "eclip-binding-k562": {
        "dataset": "eclip-binding-k562",
        "task": "classification",
        "target_col": eCLIP_K562_RBPS_LIST,
        "col_name": "eclip-binding-k562",
        "split_type": "homology",
    },
    "eclip-binding-hepg2": {
        "dataset": "eclip-binding-hepg2",
        "task": "classification",
        "target_col": eCLIP_HepG2_RBPS_LIST,
        "col_name": "eclip-binding-hepg2",
        "split_type": "homology",
    },
    "go-mf": {
        "dataset": "go-mf",
        "task": "multilabel",
        "target_col": "target",
        "col_name": "go-mf",
        "split_type": "homology",
    },
    "mrl-sugimoto": {
        "dataset": "mrl-sugimoto",
        "task": "reg_ridge",
        "target_col": "target",
        "col_name": "mrl-sugimoto",
        "split_type": "homology",
    },
    "mrl-sample-egfp-m1pseudo": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "mrl_egfp_m1pseudo",
        "col_name": "mrl-sample-egfp-m1pseudo",
        "split_type": "default",
    },
    "mrl-sample-egfp-pseudo": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "mrl_egfp_pseudo",
        "col_name": "mrl-sample-egfp-pseudo",
        "split_type": "default",
    },
    "mrl-sample-egfp-unmod": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "mrl_egfp_unmod",
        "col_name": "mrl-sample-egfp-unmod",
        "split_type": "default",
    },
    "mrl-sample-mcherry": {
        "dataset": "mrl-sample-mcherry",
        "task": "reg_ridge",
        "target_col": "mrl_mcherry",
        "col_name": "mrl-sample-mcherry",
        "split_type": "default",
    },
    "mrl-sample-designed": {
        "dataset": "mrl-sample-designed",
        "task": "reg_ridge",
        "target_col": "mrl_designed",
        "col_name": "mrl-sample-designed",
        "split_type": "default",
    },
    "mrl-sample-varying": {
        "dataset": "mrl-sample-varying",
        "task": "reg_ridge",
        "target_col": "mrl_varying_length",
        "col_name": "mrl-sample-varying",
        "split_type": "default",
    },
    "prot-loc": {
        "dataset": "prot-loc",
        "task": "multilabel",
        "target_col": "target",
        "col_name": "prot-loc",
        "split_type": "homology",
    },
    "rnahl-human": {
        "dataset": "rnahl-human",
        "task": "reg_ridge",
        "target_col": "target",
        "col_name": "rnahl-human",
        "split_type": "homology",
    },
    "rnahl-mouse": {
        "dataset": "rnahl-mouse",
        "task": "reg_ridge",
        "target_col": "target",
        "col_name": "rnahl-mouse",
        "split_type": "homology",
    },
    "rna-loc-ietswaart": {
        "dataset": "rna-loc-ietswaart",
        "task": "multilabel",
        "target_col": "target",
        "col_name": "rna-loc-ietswaart",
        "split_type": "homology",
    },
}

for ttype in ["pcg", "lncrna"]:
    split_type = "homology" if ttype == "pcg" else "ss"
    for cell in ["hap1", "hek293ft", "k562", "mda-mb-231", "thp1", "shared"]:
        # Use uppercase for target column suffix (adjust if needed).
        cell_upper = cell.upper()

        DATASET_INFO[f"{ttype}-ess-{cell}"] = {
            "dataset": f"{ttype}-ess-{cell}",
            "task": "classification",
            "target_col": f"essential_{cell_upper}",
            "col_name": f"{ttype}-ess-{cell}",
            "split_type": split_type,
        }

        if cell != "shared":
            DATASET_INFO[f"{ttype}-ess-{cell}-day14-log2fc"] = {
                "dataset": f"{ttype}-ess-{cell}",
                "task": "reg_ridge",
                "target_col": f"day14_log2fc_{cell_upper}",
                "col_name": f"{ttype}-ess-{cell}-day14-log2fc",
                "split_type": split_type,
            }
