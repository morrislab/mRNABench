from collections.abc import Callable

from .benchmark_dataset import BenchmarkDataset
from .go_bio_proc import GOBiologicalProcess
from .go_cell_comp import GOCellularComponent
from .go_mol_func import GOMolecularFunction
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .rna_loc_fazal import RNALocalizationFazal
from .rna_lifecycle_ietswaart import RNALifecycleIetswaart
from .mrl_hl_lbkwk import MRLHLLBKWK
from .prot_loc import ProteinLocalization
from .mrl_sugimoto import MRLSugimoto
from .mrl_sample import (
    MRLSampleEGFP,
    MRLSampleMCherry,
    MRLSampleDesigned,
    MRLSampleVarying
)
from .vep_traitgym import VEPTraitGymComplex, VEPTraitGymMendelian
from .mirna_target import MiRNATarget, MIRNA_TARGETS_WITH_PREFIX
from .translation_efficiency_human import TranslationEfficiencyHuman
from .translation_efficiency_mouse import TranslationEfficiencyMouse

from .eclip_binding import (
    eCLIPBindingK562,
    eCLIP_K562_TOP_RBPS_LIST,
    eCLIPBindingHepG2,
    eCLIP_HepG2_TOP_RBPS_LIST
)

DATASET_CATALOG: dict[str, Callable[..., BenchmarkDataset]] = {
    "eclip-binding-k562": eCLIPBindingK562,
    "eclip-binding-hepg2": eCLIPBindingHepG2,
    "go-bp": GOBiologicalProcess,
    "go-cc": GOCellularComponent,
    "go-mf": GOMolecularFunction,
    "rnahl-human": RNAHalfLifeHuman,
    "rnahl-mouse": RNAHalfLifeMouse,
    "rna-loc-fazal": RNALocalizationFazal,
    "rna-lifecycle-ietswaart": RNALifecycleIetswaart,
    "prot-loc": ProteinLocalization,
    "mrl-hl-lbkwk": MRLHLLBKWK,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
    "vep-traitgym-complex": VEPTraitGymComplex,
    "vep-traitgym-mendelian": VEPTraitGymMendelian,
    "mirna-target": MiRNATarget,
    "translation-efficiency-human": TranslationEfficiencyHuman,
    "translation-efficiency-mouse": TranslationEfficiencyMouse,
}

DATASET_INFO = {
    "eclip-binding-k562": {
        "dataset": "eclip-binding-k562",
        "task": "classification",
        "target_col": eCLIP_K562_TOP_RBPS_LIST,
        "split_type": "homology",
    },
    "eclip-binding-hepg2": {
        "dataset": "eclip-binding-hepg2",
        "task": "classification",
        "target_col": eCLIP_HepG2_TOP_RBPS_LIST,
        "split_type": "homology",
    },
    "go-bp": {
        "dataset": "go-bp",
        "task": "multilabel",
        "target_col": "target",
        "split_type": "homology",
    },
    "go-cc": {
        "dataset": "go-cc",
        "task": "multilabel",
        "target_col": "target",
        "split_type": "homology",
    },
    "go-mf": {
        "dataset": "go-mf",
        "task": "multilabel",
        "target_col": "target",
        "split_type": "homology",
    },
    "mrl-hl-lbkwk-hl": {
        "dataset": "mrl-hl-lbkwk",
        "task": "reg_ridge",
        "target_col": "target_in_cell_half_life",
        "split_type": "default",
    },
    "mrl-hl-lbkwk-mrl": {
        "dataset": "mrl-hl-lbkwk",
        "task": "reg_ridge",
        "target_col": "target_ribosome_load",
        "split_type": "default",
    },
    "mrl-sugimoto": {
        "dataset": "mrl-sugimoto",
        "task": "reg_ridge",
        "target_col": "target",
        "split_type": "homology",
    },
    "mrl-sample-egfp-m1pseudo": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "target_mrl_egfp_m1pseudo",
        "split_type": "default",
    },
    "mrl-sample-egfp-pseudo": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "target_mrl_egfp_pseudo",
        "split_type": "default",
    },
    "mrl-sample-egfp-unmod": {
        "dataset": "mrl-sample-egfp",
        "task": "reg_ridge",
        "target_col": "target_mrl_egfp_unmod",
        "split_type": "default",
    },
    "mrl-sample-mcherry": {
        "dataset": "mrl-sample-mcherry",
        "task": "reg_ridge",
        "target_col": "target_mrl_mcherry",
        "split_type": "default",
    },
    "mrl-sample-designed": {
        "dataset": "mrl-sample-designed",
        "task": "reg_ridge",
        "target_col": "target_mrl_designed",
        "split_type": "default",
    },
    "mrl-sample-varying": {
        "dataset": "mrl-sample-varying",
        "task": "reg_ridge",
        "target_col": "target_mrl_varying_length",
        "split_type": "default",
    },
    "prot-loc": {
        "dataset": "prot-loc",
        "task": "multilabel",
        "target_col": "target",
        "split_type": "homology",
    },
    "rnahl-human": {
        "dataset": "rnahl-human",
        "task": "reg_ridge",
        "target_col": "target",
        "split_type": "homology",
    },
    "rnahl-mouse": {
        "dataset": "rnahl-mouse",
        "task": "reg_ridge",
        "target_col": "target",
        "split_type": "homology",
    },
    "rna-loc-fazal": {
        "dataset": "rna-loc-fazal",
        "task": "multilabel",
        "target_col": "target",
        "split_type": "homology",
    },
    "rna-lifecycle-ietswaart": {
        "dataset": "rna-lifecycle-ietswaart",
        "task": "multilabel",
        "target_col": "target",
        "split_type": "homology",
    },
    "vep-traitgym-complex": {
        "dataset": "vep-traitgym-complex",
        "task": "classification",
        "target_col": "target",
        "split_type": "homology",
    },
    "vep-traitgym-mendelian": {
        "dataset": "vep-traitgym-mendelian",
        "task": "classification",
        "target_col": "target",
        "split_type": "homology",
    },
    "mirna-target": {
        "dataset": "mirna-target",
        "task": "multilabel",
        "target_col": MIRNA_TARGETS_WITH_PREFIX,
        "split_type": "homology",
    },
    "translation-efficiency-human": {
        "dataset": "translation-efficiency-human",
        "task": "reg_ridge",
        "target_col": "target",
        "split_type": "homology",
    },
    "translation-efficiency-mouse": {
        "dataset": "translation-efficiency-mouse",
        "task": "reg_ridge",
        "target_col": "target",
        "split_type": "homology",
    },
}
