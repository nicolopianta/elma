"""
Configuration of a DEIS experiment, independent of any GUI.

Two dictionaries are used:

* the **settings** -- what a user edits and saves (JSON, "version": 1, as written by the elma GUI):
  experiment / potentiostat / sequence / software_limits / awg / oscilloscope / online_analysis,
  with human-readable values ("5 V", "100 µA", "±1 V", "15 bit", "us", ...);
* the **configuration** -- the same experiment with every value resolved for hardware
  construction (SDK names, the AWG amplitude in volts, the scope current conversion factor, the
  capture size, the analysis window, ...). `resolve_configuration(settings, design)` turns the first
  into the second and elma.builder.build_run(configuration, ...) builds the run from it.

The sequence is a list of step dicts ("specs"), each with a "type" in CA, CALim, CP, CPLim, OCV,
PEIS, GEIS, Loop and the fields of that technique; every non-Loop step also carries
"deis": bool -- DEIS (AWG perturbation + oscilloscope analysis) is performed during that step.
Steps without the key get the default of their type (see spec_deis_enabled).
"""
import numpy as np

# ---------------------------------------------------------------- option labels <-> SDK names
RESOLUTION_OPTIONS = ["8 bit", "12 bit", "14 bit", "15 bit", "16 bit"]

TIME_UNIT_OPTIONS = ["fs", "ps", "ns", "us", "ms", "s"]
TIME_UNIT_TO_PICOSDK = {
    "fs": "PS5000A_FS", "ps": "PS5000A_PS", "ns": "PS5000A_NS",
    "us": "PS5000A_US", "ms": "PS5000A_MS", "s": "PS5000A_S",
}
TIME_UNIT_TO_SECONDS = {
    "fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0,
}

FREQ_UNIT_OPTIONS = ["uHz", "mHz", "Hz", "kHz"]
FREQ_UNIT_TO_HZ = {"uHz": 1e-6, "mHz": 1e-3, "Hz": 1.0, "kHz": 1e3}

VRANGE_OPTIONS = [
    "±10 mV", "±20 mV", "±50 mV", "±100 mV", "±200 mV", "±500 mV",
    "±1 V", "±2 V", "±5 V", "±10 V", "±20 V", "±50 V",
]
VRANGE_TO_PICOSDK = {
    "±10 mV": "PS5000A_10MV", "±20 mV": "PS5000A_20MV", "±50 mV": "PS5000A_50MV",
    "±100 mV": "PS5000A_100MV", "±200 mV": "PS5000A_200MV", "±500 mV": "PS5000A_500MV",
    "±1 V": "PS5000A_1V", "±2 V": "PS5000A_2V", "±5 V": "PS5000A_5V",
    "±10 V": "PS5000A_10V", "±20 V": "PS5000A_20V", "±50 V": "PS5000A_50V",
}

E_RANGE_DISPLAY_TO_SDK = {
    "2.5 V": "E_RANGE_2_5V", "5 V": "E_RANGE_5V", "10 V": "E_RANGE_10V", "Auto": "E_RANGE_AUTO",
}
E_RANGE_OPTIONS = list(E_RANGE_DISPLAY_TO_SDK.keys())

I_RANGE_DISPLAY_TO_SDK = {
    "Keep": "I_RANGE_KEEP", "100 pA": "I_RANGE_100pA", "1 nA": "I_RANGE_1nA",
    "10 nA": "I_RANGE_10nA", "100 nA": "I_RANGE_100nA", "1 µA": "I_RANGE_1uA",
    "10 µA": "I_RANGE_10uA", "100 µA": "I_RANGE_100uA", "1 mA": "I_RANGE_1mA",
    "10 mA": "I_RANGE_10mA", "100 mA": "I_RANGE_100mA", "1 A": "I_RANGE_1A",
    "Booster": "I_RANGE_BOOSTER", "Auto": "I_RANGE_AUTO",
}
I_RANGE_OPTIONS = list(I_RANGE_DISPLAY_TO_SDK.keys())
# Full-scale current of each named I range, in A -- per the EC-Lab Development Package manual
# (8.3, external control options): for galvanostatic techniques, the AWG's injected
# external-control voltage is converted to current with "1V as the full scale of the selected
# current range". Keep/Booster/Auto have no single fixed full-scale value, so current-amplitude
# conversion isn't defined for them.
I_RANGE_FULL_SCALE_AMPS = {
    "100 pA": 100e-12, "1 nA": 1e-9, "10 nA": 10e-9, "100 nA": 100e-9,
    "1 µA": 1e-6, "10 µA": 10e-6, "100 µA": 100e-6, "1 mA": 1e-3,
    "10 mA": 10e-3, "100 mA": 100e-3, "1 A": 1.0,
}

BANDWIDTH_DISPLAY_TO_SDK = {str(i): f"BW_{i}" for i in range(1, 10)}
BANDWIDTH_OPTIONS = list(BANDWIDTH_DISPLAY_TO_SDK.keys())

LIMIT_TYPE_OPTIONS = ["Ewe", "I"]
LIMIT_SIGN_OPTIONS = [">", "<"]
LIMIT_LOGIC_OPTIONS = ["and", "or"]
CONDITION_QUANTITY_OPTIONS = ["Ewe", "Ece", "I", "ElapsedTime"]

# Technique spec "type"s that carry a "duration" key (everything but Loop).
DURATION_STEP_TYPES = {"CA", "CALim", "CP", "CPLim", "OCV", "PEIS", "GEIS"}
# Per-step "deis" flag (spec["deis"]): when on, the AWG multisine runs and the oscilloscope data are
# analysed/saved during that step; when off, the AWG is silent and the scope data are discarded.
# Steps that don't use the AWG perturbation as such default to off; settings saved before the flag
# existed have no key, so the same defaults apply to them.
DEIS_OFF_BY_DEFAULT_TYPES = {"OCV", "PEIS", "GEIS"}


def spec_deis_enabled(spec: dict) -> bool:
    if spec["type"] == "Loop":
        return False
    return bool(spec.get("deis", spec["type"] not in DEIS_OFF_BY_DEFAULT_TYPES))


def deis_step_indexes(sequence) -> list:
    """Positions (EC-Lab technique indexes) in `sequence` of the steps that run DEIS."""
    return [i for i, spec in enumerate(sequence) if spec_deis_enabled(spec)]


def resolution_to_picosdk(label: str) -> str:
    bits = label.split()[0]
    return f"PS5000A_DR_{bits}BIT"


def sampling_time_seconds(value: float, unit: str) -> float:
    return value * TIME_UNIT_TO_SECONDS[unit]


# ---------------------------------------------------------------- analysis window / capture size
def compute_analysis_window_and_buffer(frequencies, sequence, sampling_time_s):
    """
    The online FFT-EIS analysis window is 1x the period of the lowest active multisine frequency --
    exactly enough for the per-window FFT-EIS to resolve it, no more. It also sets how often a new
    block (FFT-EIS + decimation of the same window) appears: window_size samples take
    window_size * sampling_time_s seconds to accumulate. Matches a validated reference script
    (time_window = 1/f_min, e.g. 100 s windows for a 0.01 Hz design); 3x the period tripled the wait
    with no benefit.

    buffer_duration (retention of the decimated voltage_ds/current_ds, i.e. what voltage.npy /
    current.npy end up holding) is the sum of the DEIS-flagged timed steps' durations, so the saved
    decimated data covers the whole run instead of a rolling slice; 3x the period if no flagged step
    has a duration.

    Returns (window_size_samples, buffer_duration_seconds), or (None, None) without frequencies.
    """
    if frequencies is None:
        return None, None
    f_min = float(np.min(frequencies))
    if f_min <= 0:
        return None, None
    period_s = 1.0 / f_min
    window_size = max(1, int(round(period_s / sampling_time_s)))
    total_duration = sum(
        spec["duration"] for spec in sequence
        if spec["type"] in DURATION_STEP_TYPES and spec_deis_enabled(spec)
    )
    buffer_duration = total_duration if total_duration > 0 else 3.0 * period_s
    return window_size, buffer_duration


# PEIS/GEIS record the initial hold (process 0) and the instrument stores every point of it in its own memory. "Record
# every dT" = 0 or "record every dI/dE" = 0 mean "record every sample" (one point per 24 us, ~40 000 rows/s): the
# memory fills within ~1 s, and the frequency-sweep results of a sweep that is short compared with the hold are lost
# (measured on the two-RC cell: GEIS 1 kHz -> 100 Hz, 4 points, wait 1 period, 0/0 -> no point at all; 0.1 s / 1 V
# -> 4 points; any positive dT AND a non-zero dI/dE are needed). The hold data is dropped by elma anyway.
PEIS_GEIS_DEFAULT_RECORD_DT = 0.1      # s
PEIS_GEIS_RECORD_NEVER = 1.0           # A / V: a change this large never happens, so only dT triggers a record


def peis_geis_recording(spec: dict):
    """(record_dt, record_dI or record_dE, corrected) of a PEIS/GEIS step, with zeros replaced by the defaults above."""
    key = "record_dI" if spec["type"] == "PEIS" else "record_dE"
    record_dt, record_d = float(spec["record_dt"]), float(spec[key])
    corrected = record_dt <= 0 or record_d <= 0
    if record_dt <= 0:
        record_dt = PEIS_GEIS_DEFAULT_RECORD_DT
    if record_d <= 0:
        record_d = PEIS_GEIS_RECORD_NEVER
    return record_dt, record_d, corrected


def step_passes(sequence) -> list:
    """
    How many times the instrument runs each step of `sequence`, Loop steps included. A Loop step with
    repeat_N = N sends the instrument back to step loop_start N more times, so the steps from loop_start
    up to the Loop run N + 1 times (nested loops multiply). A Loop that does not point back to an earlier
    step is ignored.
    """
    passes = [1] * len(sequence)
    for i, spec in enumerate(sequence):
        if spec["type"] != "Loop":
            continue
        start, extra = int(spec["loop_start"]), int(spec["repeat_N"])
        if 0 <= start < i and extra > 0:
            for j in range(start, i):
                passes[j] *= extra + 1
    return passes


def compute_capture_size(frequencies, sequence, decimation_enabled, sampling_time_s):
    """
    Oscilloscope capture size in samples. Decimation on: 3x the analysis window (the ring buffer must
    hold MORE than one window or a block can never be popped). Decimation off: the whole run (sum of
    the timed steps' durations, each counted as many times as the Loop steps make it run), captured and
    saved raw in one piece. None if it cannot be computed yet.
    """
    if decimation_enabled:
        window_size, _ = compute_analysis_window_and_buffer(frequencies, sequence, sampling_time_s)
        if window_size is None:
            return None
        return 3 * window_size
    total_duration = sum(
        spec["duration"] * passes
        for spec, passes in zip(sequence, step_passes(sequence)) if spec["type"] in DURATION_STEP_TYPES
    )
    if total_duration <= 0:
        return None
    return max(1, int(round(total_duration / sampling_time_s)))


# ---------------------------------------------------------------- AWG amplitude / scope current scale
def resolve_awg_amplitude_volts(sequence, amplitude_type, amplitude):
    """
    Returns (amplitude_volts, error_message); error_message is None when amplitude_volts is usable,
    otherwise amplitude_volts is None.

    "Potential (V)" is used as-is -- the AWG's injected voltage simply adds to the technique's own
    voltage waveform for potentiostatic techniques. "Current (A)" converts through an I range: 1 V of
    injected voltage = full scale of that I range (EC-Lab Development Package manual, 8.3), and only
    means anything for a galvanostatic (CP/CPLim) step. The whole sequence is searched, DEIS-flagged
    steps first.
    """
    if amplitude_type == "Potential (V)":
        return amplitude, None
    if not sequence:
        return None, (
            "Current amplitude needs a Chrono-Potentiometry (or w/ Limit) step somewhere "
            "in the sequence to know the I range to convert through."
        )
    galvanostatic = [s for s in sequence if s["type"] in ("CP", "CPLim")]
    first = next((s for s in galvanostatic if spec_deis_enabled(s)), None) or next(iter(galvanostatic), None)
    if first is None:
        return None, (
            "Current amplitude only makes sense for a galvanostatic step -- the AWG's "
            "injected voltage is only converted to current in that mode (it simply adds "
            "to the voltage waveform otherwise) -- no Chrono-Potentiometry (or w/ Limit) "
            "step is in the sequence."
        )
    i_range_label = first["i_range"]
    full_scale = I_RANGE_FULL_SCALE_AMPS.get(i_range_label)
    if full_scale is None:
        return None, (
            f"Can't convert current to voltage for I range '{i_range_label}' -- it doesn't "
            "have a single fixed full-scale value (Auto/Keep/Booster). Pick a specific "
            "numeric I range on the first (Chrono-Potentiometry) sequence step."
        )
    amplitude_volts = amplitude / full_scale
    if abs(amplitude_volts) > 1.0:
        return None, (
            f"{amplitude} A peak-to-peak needs {amplitude_volts:.4g} V of external-control "
            f"injection at I range '{i_range_label}' -- that exceeds the EC-Lab manual's "
            "documented +/-1V limit for galvanostatic external control. Reduce the "
            "amplitude or pick a larger I range."
        )
    return amplitude_volts, None


def resolve_scope_conv_factor(sequence, vref):
    """
    Returns (conv_factor_A_per_V, error_message). The scope's channel B (current monitor) conversion
    factor is proportional to the active I range's full-scale current: the monitor output behaves as
    a resistor R = Vref / I_full_scale, so conv_factor = I_full_scale / Vref. Taken from the first
    step that has an I range (DEIS-flagged steps first -- those are the ones the scope records).
    """
    if not sequence:
        return None, "Current conversion factor needs a sequence step with an I range to scale from."
    with_range = [s for s in sequence if s.get("i_range") is not None]
    first = next((s for s in with_range if spec_deis_enabled(s)), None) or next(iter(with_range), None)
    if first is None:
        return None, (
            "No sequence step has an I range to scale the current conversion factor from "
            "-- add a CA/CALim/CP/CPLim/GEIS step somewhere in the sequence."
        )
    i_range_label = first["i_range"]
    full_scale = I_RANGE_FULL_SCALE_AMPS.get(i_range_label)
    if full_scale is None:
        return None, (
            f"Can't scale the current conversion factor for I range '{i_range_label}' -- it "
            "doesn't have a single fixed full-scale value (Auto/Keep/Booster). Pick a "
            "specific numeric I range on the first sequence step."
        )
    return full_scale / vref, None


# ---------------------------------------------------------------- multisine design
def design_frequencies(multisine=None, multisine_split=None):
    """Combined excitation frequency set of a loaded design (single, or the low and high band of a
    split one), sorted for a split; None if nothing is loaded."""
    if multisine_split is not None:
        return np.sort(np.concatenate([multisine_split["low"].frequencies, multisine_split["high"].frequencies]))
    if multisine is not None:
        return multisine.frequencies
    return None


def load_design(path, base_frequency_hz):
    """
    Load a multisine design JSON (single or split) and rescale it so that its lowest (first
    harmonic) frequency -- the low band's, for a split -- becomes `base_frequency_hz`; every other
    frequency scales the same way, and both bands of a split by the same factor.
    Returns (multisine, multisine_split): one of them is None.
    """
    from elma.design import (
        load_multisine_json,
        load_split_multisine_bands,
        multisine_from_dict,
        rescale_multisine_frequency,
    )

    data = load_multisine_json(path)
    if data.get("type", "single") == "split":
        # load_split_multisine_bands, not multisine_from_dict directly: the high band's number_points
        # is NOT the default fs/f0 (see elma.design.io)
        low, high = load_split_multisine_bands(data)
        reference = low.frequencies.min()
        return None, {
            "low": rescale_multisine_frequency(low, base_frequency_hz, reference),
            "high": rescale_multisine_frequency(high, base_frequency_hz, reference),
        }
    multisine = multisine_from_dict(data)
    return rescale_multisine_frequency(multisine, base_frequency_hz, multisine.frequencies.min()), None


def base_frequency_hz_from_settings(settings) -> float:
    awg = settings.get("awg", {})
    return awg.get("base_frequency_value", 1.0) * FREQ_UNIT_TO_HZ[awg.get("base_frequency_unit", "Hz")]


def load_design_from_settings(settings):
    """Load the multisine named by settings["awg"]["multisine_path"] at the settings' base frequency.
    Returns (multisine, multisine_split, base_frequency_hz); (None, None, base) without a path."""
    base = base_frequency_hz_from_settings(settings)
    path = settings.get("awg", {}).get("multisine_path")
    if not path:
        return None, None, base
    multisine, split = load_design(path, base)
    return multisine, split, base


# ---------------------------------------------------------------- settings -> configuration
def resolve_configuration(settings, multisine=None, multisine_split=None):
    """
    Settings dict (see the module docstring) -> configuration dict for elma.builder.build_run.
    `multisine` / `multisine_split` are the loaded design (see load_design_from_settings); with none
    loaded, everything that depends on the design (capture size, analysis window, AWG amplitude in
    current mode...) is None or carries an error message, exactly as in the GUI.
    """
    experiment = settings.get("experiment", {})
    pot = settings.get("potentiostat", {})
    awg = settings.get("awg", {})
    scope = settings.get("oscilloscope", {})
    online = settings.get("online_analysis", {})
    sequence = list(settings.get("sequence", []))
    frequencies = design_frequencies(multisine, multisine_split)
    loaded = multisine is not None or multisine_split is not None

    scope_enabled = scope.get("enabled", True)
    decimation_enabled = scope_enabled and online.get("decimation_enabled", True)
    sampling_time_s = sampling_time_seconds(scope["sampling_value"], scope["sampling_unit"])
    amplitude_volts, amplitude_error = resolve_awg_amplitude_volts(
        sequence, awg.get("amplitude_type", "Potential (V)"), awg.get("amplitude_input", 0.0))
    if scope_enabled:
        capture_size = compute_capture_size(frequencies, sequence, decimation_enabled, sampling_time_s)
        window_size, buffer_duration = compute_analysis_window_and_buffer(frequencies, sequence, sampling_time_s)
        conv_factor, conv_factor_error = resolve_scope_conv_factor(sequence, scope.get("conv_factor_vref", 1.0))
    else:
        # None of these configure anything when the scope itself is off.
        capture_size = window_size = buffer_duration = None
        conv_factor = conv_factor_error = None
    return {
        "experiment": {
            "saving_directory": experiment["saving_directory"],
            "experiment_name": experiment["experiment_name"],
        },
        "potentiostat": {
            "address": pot["address"],
            "eclabsdk_path": pot["eclabsdk_path"],
            "channel": pot["channel"],
            "e_range": E_RANGE_DISPLAY_TO_SDK[pot["e_range"]],
            "live_plot": pot.get("live_plot", False),
            "external_control": pot.get("external_control", False),
        },
        "sequence": sequence,
        "software_limits": list(settings.get("software_limits", [])),
        "awg": {
            "enabled": awg.get("enabled", False),
            "address": awg.get("address", ""),
            "channel": awg.get("channel", 1),
            "amplitude_type": awg.get("amplitude_type", "Potential (V)"),
            "amplitude_input": awg.get("amplitude_input", 0.0),
            "amplitude_pp": amplitude_volts,
            "amplitude_error": amplitude_error,
            "base_frequency_hz": base_frequency_hz_from_settings(settings) if loaded else None,
            "multisine_loaded": loaded,
            "multisine_is_split": multisine_split is not None,
            "multisine_n_harmonics": int(frequencies.size) if frequencies is not None else None,
        },
        "oscilloscope": {
            "enabled": scope_enabled,
            "model": scope["model"],
            "resolution": resolution_to_picosdk(scope["resolution"]),
            "capture_size": capture_size,
            "sampling_time_value": scope["sampling_value"],
            "sampling_time_unit_label": scope["sampling_unit"],
            "sampling_time_unit": TIME_UNIT_TO_PICOSDK[scope["sampling_unit"]],
            "sampling_time_seconds": sampling_time_s,
            "range_a": VRANGE_TO_PICOSDK[scope["range_a"]],
            "range_b": VRANGE_TO_PICOSDK[scope["range_b"]],
            "current_conversion_factor_vref": scope.get("conv_factor_vref", 1.0),
            "current_conversion_factor": conv_factor,
            "current_conversion_factor_error": conv_factor_error,
        },
        "online_analysis": {
            "enabled": decimation_enabled,
            "window_size": window_size,
            "filter_cutoff": online.get("filter_cutoff", 10.0),
            "filter_order": online.get("filter_order", 25),
            "resampling_frequency": str(online.get("resampling_frequency", "50")),
            "buffer_duration": buffer_duration,
        },
    }
