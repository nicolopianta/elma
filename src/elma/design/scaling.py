"""
Real-Hz rescaling of a multisine designed in harmonic (dimensionless)
units.

The Multisine Designer tab's "Auto (IMD-safe)" mode designs at first
harmonic = 1 -- the actual scale factor mapping harmonic number to real Hz
is deliberately deferred to wherever the design is actually driven onto
hardware (see multisine_tab.py's module docstring), since IMD-safety,
crest factor, and waveform shape are all scale-invariant: designing once
at first-harmonic = 1 and rescaling later is equivalent to (and far
cheaper than) re-running generation at a different absolute frequency.

Rescaling a Multisine by a factor k means: frequencies *= k and
sampling_frequency *= k, with amplitudes/phases/number_points unchanged.
This preserves FFT bin alignment automatically -- bin = freq*N/fs is
invariant when freq and fs are scaled by the same factor -- so the
rescaled waveform is exact via the same FFT construction
(elma.design.waveform.compute_multisine_fft), not an
approximation, and cheap even for a 100M-sample design (today's FFT
integration made a fresh Multisine(...) construction ~1-3s at that scale).
"""
from multisine import Multisine


def rescale_multisine_frequency(multisine_obj, base_frequency_hz, reference_frequency):
    """
    Returns a NEW Multisine with every frequency and the sampling rate
    scaled by base_frequency_hz / reference_frequency -- remapping
    `reference_frequency` (typically the design's own lowest/first-harmonic
    frequency) to `base_frequency_hz`, with every other frequency scaled
    proportionally. amplitudes, phases, and number_points are carried over
    unchanged (number_points is passed through explicitly since a split
    design's high band overrides it to a non-default value -- see
    elma.design.io).

    For a multi-band design (e.g. a low/high split), pass the SAME
    reference_frequency to every band (the design's overall first-harmonic
    frequency, not each band's own minimum) so the whole design scales
    consistently and each band's frequencies stay correctly related to one
    another.
    """
    if reference_frequency <= 0:
        raise ValueError(f"reference_frequency must be positive, got {reference_frequency!r}.")
    if base_frequency_hz <= 0:
        raise ValueError(f"base_frequency_hz must be positive, got {base_frequency_hz!r}.")
    scale = base_frequency_hz / reference_frequency
    return Multisine(
        multisine_obj.sampling_frequency * scale,
        multisine_obj.frequencies * scale,
        multisine_obj.amplitudes,
        phases=multisine_obj.phases,
        number_points=multisine_obj.number_points,
    )
