from .jacobian import block_map, exact_jacobian, singular_values, top_singular_values
from .governor import PrefixVerdict, PromptCost, select_prefix, prompt_cost
from .bridge import CANONICAL_SEED, jl_matrix, project, jl_distortion_bound, comparable_ratio
from .theorems import (
    orbit_error_bound,
    pooling_recovery_bound,
    path_integral_change,
    volume_decay_rate,
    winding_witness,
)
from .repair import NEUTRAL_PREFIX, RepairReport, repair_by_context, sweep_prefixes
from .regime import RelationSpec, OrbitReport, orbit_partition, symmetry_scores
from .attractor import kaplan_yorke_dimension, embedding_bound, metric_entropy, spectrum_report
from .cocycle import lyapunov_spectrum, finite_time_spectrum
from .oseledets import growth_filtration, filtration_entropy, tolerance_sweep
from .detect import auroc, auroc_ci, MahalanobisScorer, PCAScorer
from .spectrum import BULK, log_volume, sigma_max, stable_rank, summarize, tail_alpha
from .stats import permutation_auroc, holm

__version__ = "0.1.0"

__all__ = [
    "prompt_cost",
    "select_prefix",
    "PromptCost",
    "PrefixVerdict",
    "comparable_ratio",
    "jl_distortion_bound",
    "project",
    "jl_matrix",
    "CANONICAL_SEED",
    "sweep_prefixes",
    "repair_by_context",
    "RepairReport",
    "NEUTRAL_PREFIX",
    "winding_witness",
    "volume_decay_rate",
    "path_integral_change",
    "pooling_recovery_bound",
    "orbit_error_bound",
    "RelationSpec",
    "OrbitReport",
    "orbit_partition",
    "symmetry_scores",
    "kaplan_yorke_dimension",
    "embedding_bound",
    "metric_entropy",
    "spectrum_report",
    "lyapunov_spectrum",
    "finite_time_spectrum",
    "growth_filtration",
    "filtration_entropy",
    "tolerance_sweep",
    "auroc",
    "auroc_ci",
    "MahalanobisScorer",
    "PCAScorer",
    "block_map",
    "exact_jacobian",
    "singular_values",
    "top_singular_values",
    "BULK",
    "log_volume",
    "sigma_max",
    "stable_rank",
    "summarize",
    "tail_alpha",
    "permutation_auroc",
    "holm",
]
