from collections.abc import Callable

from .benchmark_dataset import BenchmarkDataset
from .go_bio_proc import GOBiologicalProcess
from .go_cell_comp import GOCellularComponent
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
    # LNCRNAEssShared
)
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .rna_loc_fazal import RNALocalizationFazal
from .rna_loc_ietswaart import RNALocalizationIetswaart
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
    # "lncrna-ess-shared": LNCRNAEssShared,
    "rnahl-human": RNAHalfLifeHuman,
    "rnahl-mouse": RNAHalfLifeMouse,
    "rna-loc-fazal": RNALocalizationFazal,
    "rna-loc-ietswaart": RNALocalizationIetswaart,
    "prot-loc": ProteinLocalization,
    "mrl-hl-lbkwk": MRLHLLBKWK,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
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

for ttype in ["pcg", "lncrna"]:
    split_type = "homology" if ttype == "pcg" else "default"
    for cell in ["hap1", "hek293ft", "k562", "mda-mb-231", "thp1", "shared"]:

        cell_upper = cell.upper()

        if cell == "shared":

            if ttype == "lncrna": # we have too few positive samples
                continue

            DATASET_INFO[f"{ttype}-ess-{cell}"] = {
                "dataset": f"{ttype}-ess-{cell}",
                "task": ["classification"],
                "target_col":
                [
                    f"target_essential_{cell_upper}",
                ],
                "split_type": split_type,
            }
        else:
            DATASET_INFO[f"{ttype}-ess-{cell}"] = {
                "dataset": f"{ttype}-ess-{cell}",
                "task": ["classification", "reg_ridge"],
                "target_col":
                [
                    f"target_essential_{cell_upper}",
                    f"target_log2fc_{cell_upper}"
                ],
                "split_type": split_type,
            }

# This list defines the canonical order of target columns for aggregation scripts.
# It is programmatically generated to include all targets from DATASET_INFO
# in the order they appear.
TASK_ORDER = []  # Now a list of unique "dataset:target" strings
_seen_strings = set()
for info in DATASET_INFO.values():
    for target in info.get("target_col", []):
        unique_target_id = f"{info['dataset']}:{target}"
        if unique_target_id not in _seen_strings:
            TASK_ORDER.append(unique_target_id)
            _seen_strings.add(unique_target_id)
