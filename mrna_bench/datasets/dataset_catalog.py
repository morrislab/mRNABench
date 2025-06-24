from collections.abc import Callable

from .benchmark_dataset import BenchmarkDataset
from .go_bio_proc import GOBiologicalProcess
from .go_cell_comp import GOCellularComponent
from .go_mol_func import GOMolecularFunction
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .rna_loc_fazal import RNALocalizationFazal
from .rna_loc_ietswaart import RNALocalizationIetswaart
from .mrl_hl_lbkwk import MRLHLLBKWK
from .pal_tail_length_xiang import (
    PALTailLengthGV,
    PALTailLengthGVTomii
)
from .prot_loc import ProteinLocalization
from .mrl_sugimoto import MRLSugimoto
from .mrl_sample import (
    MRLSampleEGFP,
    MRLSampleMCherry,
    MRLSampleDesigned,
    MRLSampleVarying
)
from .utr_variants_bohn import (
    UTRVariantsBohnUTR5,
    UTRVariantsBohnUTR3
)
from .vep_traitgym import VEPTraitGymComplex, VEPTraitGymMendelian

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
    "rna-loc-ietswaart": RNALocalizationIetswaart,
    "pal-tail-length-xiang-gv": PALTailLengthGV,
    "pal-tail-length-xiang-gvtomii": PALTailLengthGVTomii,
    "prot-loc": ProteinLocalization,
    "mrl-hl-lbkwk": MRLHLLBKWK,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
    "utr-variants-bohn-utr5": UTRVariantsBohnUTR5,
    "utr-variants-bohn-utr3": UTRVariantsBohnUTR3,
    "vep-traitgym-complex": VEPTraitGymComplex,
    "vep-traitgym-mendelian": VEPTraitGymMendelian,
}

DATASET_INFO: dict[str, dict[str, str | list[str]]] = {
    "eclip-binding-k562": {
        "dataset": "eclip-binding-k562",
        "task": ["classification"] * len(eCLIP_K562_TOP_RBPS_LIST),
        "target_col": eCLIP_K562_TOP_RBPS_LIST,
        "split_type": "homology",
    },
    "eclip-binding-hepg2": {
        "dataset": "eclip-binding-hepg2",
        "task": ["classification"] * len(eCLIP_HepG2_TOP_RBPS_LIST),
        "target_col": eCLIP_HepG2_TOP_RBPS_LIST,
        "split_type": "homology",
    },
    "go-bp": {
        "dataset": "go-bp",
        "task": ["multilabel"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "go-cc": {
        "dataset": "go-cc",
        "task": ["multilabel"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "go-mf": {
        "dataset": "go-mf",
        "task": ["multilabel"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "mrl-sugimoto": {
        "dataset": "mrl-sugimoto",
        "task": ["reg_ridge"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "mrl-hl-lbkwk": {
        "dataset": "mrl-hl-lbkwk",
        "task": ["reg_ridge", "reg_ridge"],
        "target_col": ["target_in_cell_half_life", "target_ribosome_load"],
        "split_type": "default",
    },
    "mrl-sample-egfp": {
        "dataset": "mrl-sample-egfp",
        "task": ["reg_ridge"] * 3,
        "target_col": [
            "target_mrl_egfp_m1pseudo",
            "target_mrl_egfp_pseudo",
            "target_mrl_egfp_unmod"
        ],
        "split_type": "default",
    },
    "mrl-sample-mcherry": {
        "dataset": "mrl-sample-mcherry",
        "task": ["reg_ridge"],
        "target_col": ["target_mrl_mcherry"],
        "split_type": "default",
    },
    "mrl-sample-designed": {
        "dataset": "mrl-sample-designed",
        "task": ["reg_ridge"],
        "target_col": ["target_mrl_designed"],
        "split_type": "default",
    },
    "mrl-sample-varying": {
        "dataset": "mrl-sample-varying",
        "task": ["reg_ridge"],
        "target_col": ["target_mrl_varying_length"],
        "split_type": "default",
    },
    "pal-tail-length-xiang-gv": {
        "dataset": "pal-tail-length-xiang-gv",
        "task": ["reg_ridge"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "pal-tail-length-xiang-gvtomii": {
        "dataset": "pal-tail-length-xiang-gvtomii",
        "task": ["reg_ridge"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "prot-loc": {
        "dataset": "prot-loc",
        "task": ["multilabel"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "rnahl-human": {
        "dataset": "rnahl-human",
        "task": ["reg_ridge"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "rnahl-mouse": {
        "dataset": "rnahl-mouse",
        "task": ["reg_ridge"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "rna-loc-fazal": {
        "dataset": "rna-loc-fazal",
        "task": ["multilabel"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "rna-loc-ietswaart": {
        "dataset": "rna-loc-ietswaart",
        "task": ["multilabel"],
        "target_col": ["target"],
        "split_type": "homology",
    },
    "utr-variants-bohn-utr5": {
        "dataset": "utr-variants-bohn-utr5",
        "task": ["classification"],
        "target_col": ["target"],
        "split_type": "default",
    },
    "utr-variants-bohn-utr3": {
        "dataset": "utr-variants-bohn-utr3",
        "task": ["classification"],
        "target_col": ["target"],
        "split_type": "default",
    },
    "vep-traitgym-complex": {
        "dataset": "vep-traitgym-complex",
        "task": ["classification"],
        "target_col": ["target"],
        "split_type": "default",
    },
    "vep-traitgym-mendelian": {
        "dataset": "vep-traitgym-mendelian",
        "task": ["classification"],
        "target_col": ["target"],
        "split_type": "default",
    },
}
