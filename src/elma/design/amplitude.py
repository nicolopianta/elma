"""
Per-tone amplitude distributions for multisine design.

Two modes: equal amplitude across all tones (the default -- the overall
waveform is normalized to a +/-1 peak regardless, so the actual constant
used here is irrelevant to the final shape), and amplitude proportional to
the modulus of a loaded experimental impedance spectrum (higher impedance
-> higher excitation amplitude at that frequency, interpolated onto the
design's own frequency grid).
"""
import numpy as np
from multisine import amplitude_extraction_from_experiment


def equal_amplitudes(frequencies):
    return np.ones(np.asarray(frequencies).size)


def amplitudes_from_impedance(frequencies_desired, impedance_experiment, frequencies_experiment):
    """
    Per-tone amplitudes proportional to |Z(f)|, cubic-spline interpolated
    from an experimental impedance spectrum onto `frequencies_desired` via
    multisine.amplitude_extraction_from_experiment.

    That function expects its frequency axis in descending order (it flips
    to ascending internally for the spline) -- sorted here regardless of
    how the loaded arrays are ordered, so callers don't need to care.

    Frequencies outside the experimental spectrum's range are clamped to
    the nearest measured frequency (flat extrapolation) rather than trusting
    the cubic spline's polynomial extrapolation, which can swing wildly
    outside the fitted range.
    """
    frequencies_desired = np.asarray(frequencies_desired, dtype=float)
    frequencies_experiment = np.asarray(frequencies_experiment, dtype=float)
    impedance_experiment = np.asarray(impedance_experiment)

    if frequencies_experiment.size != impedance_experiment.size:
        raise ValueError(
            f"Impedance array ({impedance_experiment.size} points) and frequency "
            f"array ({frequencies_experiment.size} points) must be the same length."
        )

    order = np.argsort(frequencies_experiment)[::-1]  # descending, as the wrapped function expects
    frequencies_experiment = frequencies_experiment[order]
    impedance_experiment = impedance_experiment[order]

    clamped = np.clip(frequencies_desired, frequencies_experiment.min(), frequencies_experiment.max())
    amplitudes = amplitude_extraction_from_experiment(impedance_experiment, frequencies_experiment, clamped)
    return np.clip(amplitudes, 0, None)
