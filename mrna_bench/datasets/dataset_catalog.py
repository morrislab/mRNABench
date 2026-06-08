from .benchmark_dataset import BenchmarkDataset

from .eclip_binding import eCLIPBindingK562, eCLIPBindingHepG2
from .go_bio_proc import GOBiologicalProcess
from .go_cell_comp import GOCellularComponent
from .go_mol_func import GOMolecularFunction
from .mirna_target import MiRNATarget
from .mrl_hl_lbkwk import MRLHLLBKWK
from .mrl_sample import (
    MRLSampleEGFP,
    MRLSampleMCherry,
    MRLSampleDesigned,
    MRLSampleVarying,
)
from .mrl_sugimoto import MRLSugimoto
from .prot_loc import ProteinLocalization
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .rna_lifecycle_ietswaart import RNALifecycleIetswaart
from .rna_loc_fazal import RNALocalizationFazal
from .translation_efficiency_human import TranslationEfficiencyHuman
from .translation_efficiency_mouse import TranslationEfficiencyMouse
from .utr_variants_bohn import UTRVariantsBohnUTR5, UTRVariantsBohnUTR3
from .vep_traitgym import VEPTraitGymComplex, VEPTraitGymMendelian


DATASET_CATALOG: dict[str, type[BenchmarkDataset]] = {
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
    "mirna-target": MiRNATarget,
    "mrl-hl-lbkwk": MRLHLLBKWK,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
    "translation-efficiency-human": TranslationEfficiencyHuman,
    "translation-efficiency-mouse": TranslationEfficiencyMouse,
    "utr-variants-bohn-utr5": UTRVariantsBohnUTR5,
    "utr-variants-bohn-utr3": UTRVariantsBohnUTR3,
    "vep-traitgym-complex": VEPTraitGymComplex,
    "vep-traitgym-mendelian": VEPTraitGymMendelian,
}

DATASET_INFO: dict[str, dict[str, str | list[str] | bool]] = {
    name: {
        "dataset": cls.METADATA.dataset_name,
        "task": cls.METADATA.task,
        "target_col": cls.METADATA.target_col,
        "default_split_type": cls.METADATA.default_split_type,
        "vep": cls.METADATA.vep,
    }
    for name, cls in DATASET_CATALOG.items()
}
