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

DATASET_INFO = {
    "go-mf": {
        "dataset": "go-mf",
        "task": "multilabel",  # reported as AUROC/AUPRC
        "target_col": "target",
        "col_name": "go-mf (AUROC/AUPRC)",
        "split_type": "homology",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sugimoto": {
        "dataset": "mrl-sugimoto",
        "task": "reg_ridge",  # reported as MSE/R
        "target_col": "target",
        "col_name": "mrl-sugimoto (MSE/R)",
        "split_type": "homology",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sample-egfp-m1pseudo": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "mrl_egfp_m1pseudo",
        "col_name": "mrl-sample-egfp (m1pseudo) (MSE/R)",
        "split_type": "default",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sample-egfp-pseudo": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "mrl_egfp_pseudo",
        "col_name": "mrl-sample-egfp (pseudo) (MSE/R)",
        "split_type": "default",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sample-egfp-unmod": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "mrl_egfp_unmod",
        "col_name": "mrl-sample-egfp (unmod) (MSE/R)",
        "split_type": "default",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sample-mcherry": {
        "dataset": "mrl-sample-mcherry",
        "task": "reg_ridge",
        "target_col": "mrl_mcherry",
        "col_name": "mrl-sample-mcherry (MSE/R)",
        "split_type": "default",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sample-designed": {
        "dataset": "mrl-sample-designed",
        "task": "reg_ridge",
        "target_col": "mrl_designed",
        "col_name": "mrl-sample-designed (MSE/R)",
        "split_type": "default",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "mrl-sample-varying": {
        "dataset": "mrl-sample-varying",
        "task": "reg_ridge",
        "target_col": "mrl_varying_length",
        "col_name": "mrl-sample-varying (MSE/R)",
        "split_type": "default",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "prot-loc": {
        "dataset": "prot-loc",
        "task": "multilabel",
        "target_col": "target",
        "col_name": "prot-loc (AUROC/AUPRC)",
        "split_type": "homology",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "rnahl-human": {
        "dataset": "rnahl-human",
        "task": "reg_ridge",
        "target_col": "target",
        "col_name": "rnahl-human (MSE/R)",
        "split_type": "homology",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "rnahl-mouse": {
        "dataset": "rnahl-mouse",
        "task": "reg_ridge",
        "target_col": "target",
        "col_name": "rnahl-mouse (MSE/R)",
        "split_type": "homology",
        "isoform_resolved": False,
        "transcript_avg": False,
    },
    "lncrna-ess-isoform-level": {
        "dataset": "lncrna-ess-shared",
        "task": "classification",
        "target_col": "essential_SHARED",
        "col_name": "lncrna-ess (AUROC/AUPRC)",
        "split_type": "ss",
        "isoform_resolved": True,
        "transcript_avg": False,
    },
}
