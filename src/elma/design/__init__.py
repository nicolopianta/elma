from elma.design.waveform import compute_multisine_fft  # noqa: F401 -- import patches Multisine.compute_multisine
from elma.design.frequency_generation import (
    generate_imd_safe_frequencies,
    snap_top_frequency_to_power_of_ten,
)
from elma.design.phase_optimization import (
    optimize_phases_differential_evolution,
    optimize_phases_random_search,
)
from elma.design.io import (
    save_multisine_json,
    save_split_multisine_json,
    load_multisine_json,
    multisine_from_dict,
    load_split_multisine_bands,
)
from elma.design.amplitude import equal_amplitudes, amplitudes_from_impedance
from elma.design.splitting import split_multisine
from elma.design.scaling import rescale_multisine_frequency

__all__ = [
    "compute_multisine_fft",
    "generate_imd_safe_frequencies",
    "snap_top_frequency_to_power_of_ten",
    "optimize_phases_differential_evolution",
    "optimize_phases_random_search",
    "save_multisine_json",
    "save_split_multisine_json",
    "load_multisine_json",
    "multisine_from_dict",
    "load_split_multisine_bands",
    "equal_amplitudes",
    "amplitudes_from_impedance",
    "split_multisine",
    "rescale_multisine_frequency",
]
