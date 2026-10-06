"""
Minimal JSON persistence for a designed multisine.

Deliberately stores only what's needed to reconstruct the excitation
(sampling frequency, frequency/amplitude/phase arrays) -- not the derived
waveform, crest factor, or normalization bookkeeping that
Multisine.save_metadata writes.

Two shapes: "single" (one multisine) and "split" (a low/high band pair from
splitting.split_multisine, tagged with the split frequency). The "type"
field lets a downstream consumer -- e.g. the Experiment Builder tab --
dispatch to the right online processing (single-channel vs. dual-band
data cutting/filtering) without guessing from the file's shape.

number_points is saved explicitly (not just re-derived from sampling_
frequency/frequencies[0] on load): that default formula is only correct
for a multisine whose number_points was never overridden. split_multisine's
high band deliberately overrides it to a fixed window decoupled from its
own lowest harmonic (a short, representative segment for looping on an
AWG channel, not a bin-aligned capture -- see splitting.py) -- reconstructing
it via the default formula instead silently gives the wrong duration
(confirmed: ~105x too short on a real 7-decade split design). Loading falls
back to split_multisine's own formula for files saved before this field
existed (see load_split_multisine_bands).
"""
import json

import numpy as np
from multisine import Multisine


def _multisine_to_dict(multisine_obj):
    return {
        "sampling_frequency": float(multisine_obj.sampling_frequency),
        "frequencies": np.asarray(multisine_obj.frequencies).tolist(),
        "amplitudes": np.asarray(multisine_obj.amplitudes).tolist(),
        "phases": np.asarray(multisine_obj.phases).tolist(),
        "number_points": int(multisine_obj.number_points),
    }


def multisine_from_dict(d, number_points=None):
    """
    Reconstruct a Multisine from a dict shaped like _multisine_to_dict's
    output. Uses d["number_points"] when present (current file format);
    otherwise falls back to the explicit `number_points` argument if
    given, or Multisine's own default (sampling_frequency/frequencies[0])
    otherwise -- correct for a single design or a split's low band, WRONG
    for a split's high band (see module docstring) unless the caller
    supplies the right value (load_split_multisine_bands does this for
    older files).
    """
    kwargs = {}
    if "number_points" in d:
        kwargs["number_points"] = d["number_points"]
    elif number_points is not None:
        kwargs["number_points"] = number_points
    return Multisine(
        float(d["sampling_frequency"]),
        np.array(d["frequencies"], dtype=float),
        np.array(d["amplitudes"], dtype=float),
        phases=np.array(d["phases"], dtype=float),
        **kwargs,
    )


def load_split_multisine_bands(data):
    """
    Given data = load_multisine_json(path) for a "split" design, returns
    (low, high) Multisine instances with the correct number_points for
    each band. Backward compatible with files saved before number_points
    was included in the JSON: the high band's is recomputed via
    splitting.split_multisine's own formula (the low band needs no
    special-casing -- its default fs/f0 already matches what
    split_multisine produces).
    """
    split_frequency = float(data["split_frequency"])
    low = multisine_from_dict(data["low_band"])
    high_dict = data["high_band"]
    high_number_points = high_dict.get("number_points")
    if high_number_points is None:
        high_number_points = round(float(high_dict["sampling_frequency"]) * (100.0 / split_frequency))
    high = multisine_from_dict(high_dict, number_points=high_number_points)
    return low, high


def save_multisine_json(multisine_obj, path):
    data = {"type": "single", **_multisine_to_dict(multisine_obj)}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_split_multisine_json(low_multisine, high_multisine, split_frequency, path):
    data = {
        "type": "split",
        "split_frequency": float(split_frequency),
        "low_band": _multisine_to_dict(low_multisine),
        "high_band": _multisine_to_dict(high_multisine),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_multisine_json(path):
    """
    Returns the raw parsed JSON (dict). Callers should branch on
    data.get("type", "single") -- "single" gives sampling_frequency /
    frequencies / amplitudes / phases directly, "split" gives
    split_frequency / low_band / high_band (each shaped like the single
    case) -- and reconstruct whichever Multisine object(s) they need.
    """
    with open(path) as f:
        return json.load(f)
