"""
Split a designed multisine into a low- and a high-frequency band, e.g. for
driving two separate AWG channels. Faithful port of the original
MultisineGenerator.py script's multisine_split, adapted to operate on a
multisine.Multisine instance (and its normalize-to-+/-1-peak convention)
instead of raw arrays.

Each band gets its own sampling regime rather than sharing the parent's:

- low band: sampled at 10 * split_frequency, spanning one period of its own
  lowest harmonic (Multisine's default number_points -- this already
  matches the intent since the parent's default has the same shape).
- high band: sampled at 10 * (its own max harmonic). Its window length
  depends on how the parent's frequencies were generated:
    * If every high-band frequency is an exact integer multiple of
      split_frequency (elma.design.frequency_generation.
      generate_imd_safe_frequencies' own split_frequency argument
      guarantees this), split_frequency itself is an exact FFT
      bin-alignment reference -- one period of split_frequency
      (number_points = sampling_frequency/split_frequency) exactly
      represents every high-band frequency, no more.
    * Otherwise (a design generated without that argument, so the high
      band's frequencies share no useful common factor -- see
      frequency_generation.py), that reference doesn't hold, and the
      window falls back to a *fixed* 100/split_frequency, deliberately
      decoupled from the high band's own lowest harmonic (which could
      otherwise force an excessively long window) -- a short
      representative segment, safe for looping on an AWG but far larger
      than necessary. This is why using generate_imd_safe_frequencies'
      split_frequency argument up front matters for AWG memory: without
      it, the high band can easily need 100x more samples than with it.

Both bands reuse the corresponding slice of the parent's current
amplitudes and phases (no re-optimization), and are independently
normalized to a +/-1 peak, same as any freshly generated multisine in this
tool.
"""
import numpy as np
from multisine import Multisine


def _is_exact_multiple_of(values, reference, rtol=1e-9):
    ratios = values / reference
    return np.allclose(ratios, np.round(ratios), rtol=0, atol=rtol * max(1.0, np.max(np.abs(ratios))))


def split_multisine(multisine_obj, split_frequency):
    """
    Returns (low, high) Multisine instances. Raises ValueError if
    `split_frequency` doesn't separate the parent's harmonics into two
    non-empty bands.
    """
    if split_frequency <= 0:
        raise ValueError("Split frequency must be positive.")

    frequencies = np.asarray(multisine_obj.frequencies)
    amplitudes = np.asarray(multisine_obj.amplitudes)
    phases = np.asarray(multisine_obj.phases)

    low_mask = frequencies < split_frequency
    high_mask = ~low_mask
    if not low_mask.any() or not high_mask.any():
        raise ValueError(
            f"Split frequency {split_frequency:g} does not separate the {frequencies.size} "
            f"harmonics (range [{frequencies.min():g}, {frequencies.max():g}]) into two non-empty bands."
        )

    low = Multisine(
        10.0 * split_frequency,
        frequencies[low_mask],
        amplitudes[low_mask],
        phases=phases[low_mask],
    )
    low.normalize_waveform()

    high_freqs = frequencies[high_mask]
    high_sampling_frequency = 10.0 * high_freqs.max()
    if _is_exact_multiple_of(high_freqs, split_frequency):
        high_number_points = int(round(high_sampling_frequency / split_frequency))
    else:
        high_number_points = int(round(high_sampling_frequency * (100.0 / split_frequency)))
    high = Multisine(
        high_sampling_frequency,
        high_freqs,
        amplitudes[high_mask],
        phases=phases[high_mask],
        number_points=high_number_points,
    )
    high.normalize_waveform()

    return low, high
