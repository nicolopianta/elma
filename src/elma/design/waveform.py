"""
FFT-based multisine waveform construction, replacing
Multisine.compute_multisine's O(n_freq * n_points) direct sin-summation
loop.

DEIS multisines are unique in that every excitation frequency is, by the
project's own construction (elma.design.frequency_generation.
generate_imd_safe_frequencies), an exact integer multiple of the base
frequency -- and Multisine.number_points defaults to
sampling_frequency/frequencies[0], so each frequency f = first_freq * k
lands exactly on FFT bin k. That makes an IFFT-based construction not an
approximation but an exact equivalent: verified to ~3e-11 against the
original direct-summation implementation on real generate_imd_safe_
frequencies output, and 8-40x faster for realistic multisines -- 39x for a
7-decade, 100M-sample waveform (64s -> 1.6s).

Multisine.compute_multisine is monkey-patched below, at import time, to
use this implementation instead, so every call site benefits automatically
-- including Multisine.__init__'s own initial waveform computation and
Multisine.best_random_phases' internal loop, not just elma's own
callers. This is safe because the replacement is a verified exact
equivalent of the original (same inputs/outputs/side effects), not a
behavior change.

Bin alignment needs more than "every frequency is an integer multiple of
the base frequency" -- it needs number_points to be an integer multiple of
the fundamental period's sample count (sampling_frequency/frequencies[0]),
which is exactly what Multisine's own default (number_points=None) always
produces. elma.design.splitting.split_multisine's high band is the
one place in this project that deliberately overrides number_points to a
fixed window decoupled from its own lowest harmonic (a short, representative
segment for looping on an AWG, not a bin-aligned capture) -- so the patched
compute_multisine below falls back to direct summation specifically when
compute_multisine_fft reports misaligned bins, keeping that legitimate case
working. That array is short, so the fallback costs nothing there. The
optimizer objectives in phase_optimization.py call compute_multisine_fft
directly (no fallback) since they only ever run on freshly-generated,
inherently bin-aligned multisines -- hitting the error there would mean a
real bug, not this decoupled-window case.

multisine.compute_crest_factor and Multisine.normalize_waveform are
patched too, for the same reason: once waveform construction stopped
being the bottleneck, they were. Both compute max(abs(signal)) with
Python's builtin max() over a numpy array (unvectorized, ~5x slower than
np.max for a 100M-sample array, measured) -- compute_crest_factor once
per phase-optimization trial, normalize_waveform once per "Generate"
click (it's called unconditionally right after construction). compute_
crest_factor is a free function, referenced by bare name inside
multisine's own __init__ and best_random_phases, so patching the module
attribute (multisine.compute_crest_factor) redirects those internal call
sites too, the same way patching Multisine.compute_multisine does for a
method.
"""
import numpy as np
import multisine as _multisine_pkg
from multisine import Multisine


def compute_multisine_fft(multisine_obj, phases):
    """
    FFT-based equivalent of Multisine.compute_multisine(phases): exact
    (not approximate) whenever every frequency lands on an integer FFT bin
    for multisine_obj's number_points/sampling_frequency -- true by
    construction for every DEIS multisine this project generates, since
    frequencies are always integer multiples of the base frequency. Raises
    ValueError rather than silently returning a wrong waveform if that
    assumption is ever violated (e.g. a hand-edited design file).

    Requires multisine_obj.time to already be populated (Multisine.__init__
    and the patched compute_multisine below both guarantee this).
    """
    N = multisine_obj.time.size
    fs = multisine_obj.sampling_frequency
    frequencies = np.asarray(multisine_obj.frequencies)
    amplitudes = np.asarray(multisine_obj.amplitudes)
    phases = np.asarray(phases)

    bin_float = frequencies * N / fs
    bins = np.round(bin_float).astype(np.int64)
    max_error = np.max(np.abs(bin_float - bins)) if bins.size else 0.0
    if max_error > 1e-6:
        raise ValueError(
            f"Frequencies don't land on exact FFT bins (max error {max_error:.3g} bins) -- "
            "every DEIS multisine frequency must be an exact integer multiple of the base "
            "frequency for the FFT-based waveform construction to be exact."
        )
    if bins.size and (np.any(bins <= 0) or np.any(bins >= N // 2 + 1)):
        raise ValueError("A frequency's bin index is out of the valid rfft range for this N.")

    X = np.zeros(N // 2 + 1, dtype=complex)
    X[bins] = -1j * (amplitudes * N / 2) * np.exp(1j * phases)
    return np.fft.irfft(X, n=N)


def _direct_compute_multisine(time, frequencies, amplitudes, phases):
    # The original, unpatched algorithm -- used only as a fallback when
    # bins don't align (see module docstring: split_multisine's high band).
    waveform = np.zeros(time.size)
    for i in range(frequencies.size):
        waveform += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * time + phases[i])
    return waveform


def _patched_compute_multisine(self, phases, time_step=None):
    # Drop-in replacement for the original Multisine.compute_multisine:
    # same side effects (self.time, self.time_step) and return value. Uses
    # the FFT path whenever bins align (every DEIS multisine this project
    # generates); falls back to direct summation otherwise (see module
    # docstring).
    self.time = np.arange(self.number_points, dtype=np.float64)
    self.time /= self.sampling_frequency
    try:
        waveform = compute_multisine_fft(self, phases)
    except ValueError:
        waveform = _direct_compute_multisine(self.time, self.frequencies, self.amplitudes, phases)
    self.time_step = waveform.size / self.sampling_frequency
    return waveform


def _compute_crest_factor_fast(signal):
    # Vectorized equivalent of multisine.compute_crest_factor -- see
    # module docstring.
    signal = np.asarray(signal)
    return np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2))


def _patched_normalize_waveform(self, factor=1):
    # Drop-in replacement for the original Multisine.normalize_waveform:
    # same side effects and result, using np.max instead of Python's
    # builtin max() -- see module docstring.
    self.waveform = np.divide(self.waveform, np.max(np.abs(self.waveform)))
    self.waveform = np.multiply(self.waveform, factor)
    self.is_normalized = True
    self.normalization_factor = factor


Multisine.compute_multisine = _patched_compute_multisine
Multisine.normalize_waveform = _patched_normalize_waveform
_multisine_pkg.compute_crest_factor = _compute_crest_factor_fast
