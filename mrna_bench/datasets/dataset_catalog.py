from collections.abc import Callable

from .benchmark_dataset import BenchmarkDataset
from .go_mol_func import GOMolecularFunction
from .rna_hl_human import RNAHalfLifeHuman
from .rna_hl_mouse import RNAHalfLifeMouse
from .prot_loc import ProteinLocalization
from .mrl_sugimoto import MRLSugimoto
from .mrl_sample import (
    MRLSampleEGFP,
    MRLSampleMCherry,
    MRLSampleDesigned,
    MRLSampleVarying
)

DATASET_CATALOG: dict[str, Callable[..., BenchmarkDataset]] = {
    "go-mf": GOMolecularFunction,
    "rnahl-human": RNAHalfLifeHuman,
    "rnahl-mouse": RNAHalfLifeMouse,
    "prot-loc": ProteinLocalization,
    "mrl-sugimoto": MRLSugimoto,
    "mrl-sample-egfp": MRLSampleEGFP,
    "mrl-sample-mcherry": MRLSampleMCherry,
    "mrl-sample-designed": MRLSampleDesigned,
    "mrl-sample-varying": MRLSampleVarying,
}

# Maps dataset name to valid target cols and default task.
DATASET_DEFAULT_TASK: dict[str, dict[str, str]] = {
    "go-mf": {"target": "multilabel"},
    "rnahl-human": {"target": "regression"},
    "rnahl-mouse": {"target": "regression"},
    "prot-loc": {"target": "multilabel"},
    "mrl-sugimoto": {"target": "regression"},
    "pcg-ess": {"target": "regression"},
    "mrl-sample-mcherry": {"mrl_mcherry": "regression"},
    "mrl-sample-egfp": {
        "mrl_egfp_m1pseudo": "regression",
        "mrl_egfp_pseudo": "regression",
        "mrl_egfp_unmod": "regression"
    },
    "mrl-sample-designed": {"mrl_designed": "regression"},
    "mrl-sample-varying": {"mrl_varying_length": "regression"},
}
