"""
Build a DEIS run from a resolved configuration (see elma.config), without any GUI.

    settings = json.load(open("experiment.json"))
    multisine, split, _ = elma.config.load_design_from_settings(settings)
    config = elma.config.resolve_configuration(settings, multisine, split)
    run = elma.builder.build_run(config, multisine, split)
    run.start()          # ... run.running / elma.runs.has_finished(run) ... run.stop()

build_run returns one of
- elma.DEISchannel          oscilloscope on and online analysis on (the DEIS measurement),
- elma.runs.RawCaptureRun   oscilloscope on, online analysis off (whole-run raw capture),
- elma.runs.PotentiostatOnlyRun   oscilloscope off.
All expose start()/stop(); the first two also keep the potentiostat channel in `.potentiostat`
(a PEISAwareChannel, so PEIS/GEIS steps deliver their impedance points in `.peis_geis_points()`).

Which steps run DEIS is the per-step "deis" flag of the sequence (config.spec_deis_enabled): the AWG
is switched on for exactly those technique indexes, and only their oscilloscope data are analysed
and saved.

The hardware classes are looked up in a `Hardware` object so the wiring can be tested without
instruments; default_hardware() imports the real ones.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from elma import config as C
from elma.utils import ConditionAverage


@dataclass
class Hardware:
    BiologicDevice: Any
    ChannelConfig: Any
    FileWriter: Any
    Channel: Any            # PEISAwareChannel by default
    E_RANGE: Any
    I_RANGE: Any
    BANDWIDTH: Any
    techniques: Any         # namespace with ChronoAmperometry, ..., Loop, build_limit, generate_xctr_param
    PEISTechnique: Any
    GEISTechnique: Any
    Picoscope4000: Any
    Picoscope5000a: Any
    TrueFormAWG: Any


def default_hardware() -> Hardware:
    """The real instrument classes (pyeclab, pypicostreaming, trueformawg)."""
    import pyeclab.techniques as techniques
    from pyeclab import BANDWIDTH, E_RANGE, I_RANGE, BiologicDevice, ChannelConfig, FileWriter
    from pypicostreaming import Picoscope4000, Picoscope5000a
    from trueformawg import TrueFormAWG

    from elma.peis_geis import GEISTechnique, PEISAwareChannel, PEISTechnique

    return Hardware(
        BiologicDevice=BiologicDevice, ChannelConfig=ChannelConfig, FileWriter=FileWriter,
        Channel=PEISAwareChannel, E_RANGE=E_RANGE, I_RANGE=I_RANGE, BANDWIDTH=BANDWIDTH,
        techniques=techniques, PEISTechnique=PEISTechnique, GEISTechnique=GEISTechnique,
        Picoscope4000=Picoscope4000, Picoscope5000a=Picoscope5000a, TrueFormAWG=TrueFormAWG,
    )


def build_sequence(config: dict, device, hw: Hardware, channel_config):
    """The technique objects of config["sequence"], in order (make_technique() already called)."""
    t = hw.techniques
    # ChannelConfig.external_control alone does nothing on the instrument -- it has to be turned into
    # the xctr bitfield and passed to each technique explicitly, or "external control" is a silent no-op.
    xctr = t.generate_xctr_param(channel_config)
    # E range is one value for the whole channel/sequence, not per technique step.
    e_range = hw.E_RANGE[config["potentiostat"]["e_range"]]

    def i_range(spec):
        return hw.I_RANGE[C.I_RANGE_DISPLAY_TO_SDK[spec["i_range"]]]

    def bandwidth(spec):
        return hw.BANDWIDTH[C.BANDWIDTH_DISPLAY_TO_SDK[spec["bandwidth"]]]

    def limit(spec):
        return t.build_limit(spec["limit_type"], spec["limit_sign"], spec["limit_logic"], active=True)

    sequence = []
    for spec in config["sequence"]:
        kind = spec["type"]
        if kind == "CA":
            tech = t.ChronoAmperometry(
                device=device, voltage=spec["voltage"], duration=spec["duration"], vs_init=spec["vs_init"],
                nb_steps=spec["nb_steps"], record_dt=spec["record_dt"], record_dI=spec["record_dI"],
                repeat=spec["repeat"], e_range=e_range, i_range=i_range(spec), bandwidth=bandwidth(spec), xctr=xctr,
            )
        elif kind == "CALim":
            tech = t.ChronoAmperometryWithLimits(
                device=device, voltage=spec["voltage"], duration=spec["duration"], vs_init=spec["vs_init"],
                nb_steps=spec["nb_steps"], record_dt=spec["record_dt"], record_dI=spec["record_dI"],
                repeat=spec["repeat"], e_range=e_range, i_range=i_range(spec), limit_variable=limit(spec),
                limit_value=spec["limit_value"], bandwidth=bandwidth(spec), xctr=xctr,
            )
        elif kind == "CP":
            tech = t.ChronoPotentiometry(
                device=device, current=spec["current"], duration=spec["duration"], vs_init=spec["vs_init"],
                nb_steps=spec["nb_steps"], record_dt=spec["record_dt"], record_dE=spec["record_dE"],
                repeat=spec["repeat"], i_range=i_range(spec), e_range=e_range, bandwidth=bandwidth(spec), xctr=xctr,
            )
        elif kind == "CPLim":
            tech = t.ChronoPotentiometryWithLimits(
                device=device, current=spec["current"], duration=spec["duration"], vs_init=spec["vs_init"],
                nb_steps=spec["nb_steps"], record_dt=spec["record_dt"], record_dE=spec["record_dE"],
                repeat=spec["repeat"], i_range=i_range(spec), e_range=e_range, limit_variable=limit(spec),
                limit_value=spec["limit_value"], bandwidth=bandwidth(spec), xctr=xctr,
            )
        elif kind == "OCV":
            tech = t.OpenCircuitVoltage(
                device=device, duration=spec["duration"], record_dt=spec["record_dt"], e_range=e_range,
                bandwidth=bandwidth(spec), xctr=xctr,
            )
        elif kind in ("PEIS", "GEIS"):
            # Independent per-step xctr, NOT the sequence-wide one: PEIS/GEIS default to external
            # control OFF even when the potentiostat's own external control is on -- they exist to
            # measure through the potentiostat's own internal sine generator, and silently
            # inheriting the global xctr would defeat that.
            step_xctr = t.generate_xctr_param(hw.ChannelConfig(external_control=spec["external_control"]))
            if kind == "PEIS":
                tech = hw.PEISTechnique(
                    device=device, vs_initial=spec["vs_init"], initial_voltage_step=spec["voltage_step"],
                    duration_step=spec["duration"], record_every_dT=spec["record_dt"],
                    record_every_dI=spec["record_dI"], initial_frequency=spec["initial_frequency"],
                    final_frequency=spec["final_frequency"], sweep_linear=spec["sweep_linear"],
                    amplitude_voltage=spec["amplitude"], frequency_number=spec["frequency_number"],
                    average_n_times=spec["average_n_times"], correction=spec["correction"],
                    wait_for_steady=spec["wait_for_steady"], xctr=step_xctr,
                )
            else:
                tech = hw.GEISTechnique(
                    device=device, vs_initial=spec["vs_init"], initial_current_step=spec["current_step"],
                    duration_step=spec["duration"], record_every_dT=spec["record_dt"],
                    record_every_dE=spec["record_dE"], initial_frequency=spec["initial_frequency"],
                    final_frequency=spec["final_frequency"], sweep_linear=spec["sweep_linear"],
                    amplitude_current=spec["amplitude"], frequency_number=spec["frequency_number"],
                    average_n_times=spec["average_n_times"], correction=spec["correction"],
                    wait_for_steady=spec["wait_for_steady"], i_range=i_range(spec), xctr=step_xctr,
                )
        elif kind == "Loop":
            tech = t.Loop(device=device, repeat_N=spec["repeat_N"], loop_start=spec["loop_start"])
        else:
            raise ValueError(f"Unknown sequence step type {kind!r}")
        tech.make_technique()
        sequence.append(tech)
    return sequence


def _setup_awg_channel(hw: Hardware, address, channel_num, multisine, amplitude_pp):
    """Upload and configure one physical AWG channel; returns the TrueFormAWG device."""
    device = hw.TrueFormAWG(address, channel_num)
    device.device.write("*CLS")  # drop any stale queued errors from earlier sessions
    device.clear_ch_mem()
    # Defensive: normalize_waveform() scales the peak sample to exactly +/-1.0; clip after the
    # float64 -> float32 cast in case rounding pushes a sample marginally past the AWG's valid ARB range.
    waveform_arb = np.clip(multisine.waveform.astype("float32"), -1.0, 1.0)
    device.load_awf("multisine", waveform_arb.tolist())
    device.select_awf("multisine")
    device.set_Z_out_infinite()
    # Apply amplitude and sample rate explicitly, here: MultisineGenerator.update() only fires on a
    # technique-index *transition* detected by Channel's polling loop, which never happens for a
    # single-step sequence -- set_amplitude()/set_sample_rate() would silently never be called.
    device.set_sample_rate(multisine.sampling_frequency)
    device.set_amplitude(amplitude_pp)
    # Drain and print the FULL error queue, not just one entry -- a single query() can report a stale
    # error left over from a previous run rather than this upload.
    for _ in range(10):
        queued_error = device.device.query("SYSTEM:ERROR?").strip()
        print(f"AWG channel {channel_num} error queue: {queued_error}")
        if queued_error.startswith("+0"):
            break
    return device


def build_awg(config: dict, multisine, multisine_split, deis_indexes, hw: Hardware):
    """Returns (awg, awg_channels): the generator that follows the DEIS-flagged steps and the
    TrueFormAWG devices (for turning off), or (None, []) when the AWG is disabled."""
    from elma.multisinegen import MultisineGenerator, MultisineGeneratorCombined

    awg_cfg = config["awg"]
    if not awg_cfg["enabled"]:
        return None, []
    if awg_cfg["amplitude_error"]:
        raise ValueError(awg_cfg["amplitude_error"])
    if multisine is None and multisine_split is None:
        raise ValueError("AWG is enabled but no multisine design is loaded.")
    amplitude = awg_cfg["amplitude_pp"]
    # sequence_indexes = the DEIS-flagged steps: MultisineGenerator(Combined).update(index) -- called on
    # every technique change -- turns the AWG on for an index in this list and OFF for any other.
    if multisine_split is not None:
        # Split design: low band on AWG channel 1, high band on channel 2, digitally combined into
        # channel 1's physical output (a single cable to the cell). MultisineGeneratorCombined calls
        # TrueFormAWG.combine_channels() on channel 1 for us (its default source_channel='CH2' is only
        # correct when channel1 really is physical channel 1 and channel2 physical channel 2).
        low, high = multisine_split["low"], multisine_split["high"]
        device_low = _setup_awg_channel(hw, awg_cfg["address"], 1, low, amplitude)
        device_high = _setup_awg_channel(hw, awg_cfg["address"], 2, high, amplitude)
        awg = MultisineGeneratorCombined(
            channel1=MultisineGenerator.for_steps(device_low, deis_indexes, "multisine", low.sampling_frequency, amplitude),
            channel2=MultisineGenerator.for_steps(device_high, deis_indexes, "multisine", high.sampling_frequency, amplitude),
            waveforms_names=["multisine_low", "multisine_high"],
        )
        return awg, [device_low, device_high]
    device = _setup_awg_channel(hw, awg_cfg["address"], awg_cfg["channel"], multisine, amplitude)
    awg = MultisineGenerator.for_steps(device, deis_indexes, "multisine", multisine.sampling_frequency, amplitude)
    return awg, [device]


def build_run(config: dict, multisine=None, multisine_split=None, hardware: Optional[Hardware] = None):
    """Connect to the instruments and build the run described by `config` (see the module docstring).
    Raises ValueError for an inconsistent configuration before any scope is created."""
    hw = hardware or default_hardware()
    saving_dir = Path(config["experiment"]["saving_directory"]) / config["experiment"]["experiment_name"]
    frequencies = C.design_frequencies(multisine, multisine_split)

    device = hw.BiologicDevice(
        config["potentiostat"]["address"], binary_path=config["potentiostat"]["eclabsdk_path"],
    )
    channel_config = hw.ChannelConfig(
        # Always False: pyeclab's own LivePlot pops up a separate, blocking matplotlib window; a GUI
        # draws its own plots from measurement_data.txt.
        live_plot=False,
        external_control=config["potentiostat"]["external_control"],
    )
    sequence = build_sequence(config, device, hw, channel_config)
    writer = hw.FileWriter(
        file_dir=Path(config["experiment"]["saving_directory"]), experiment_name=config["experiment"]["experiment_name"],
    )
    channel = hw.Channel(device, config["potentiostat"]["channel"], writer=writer, config=channel_config)
    channel.load_sequence(sequence)

    conditions = [
        ConditionAverage(spec["technique_index"], spec["quantity"], spec["operator"], spec["threshold"], spec["num_elements"])
        for spec in config["software_limits"]
    ]
    deis_indexes = C.deis_step_indexes(config["sequence"])
    awg, awg_channels = build_awg(config, multisine, multisine_split, deis_indexes, hw)

    scope = config["oscilloscope"]
    if not scope["enabled"]:
        from elma.runs import PotentiostatOnlyRun
        return PotentiostatOnlyRun(channel=channel, awg_channels=awg_channels, conditions=conditions, awg=awg)

    if scope["capture_size"] is None:
        raise ValueError(
            "Could not compute oscilloscope capture size -- load a multisine design "
            "(decimation on) or add a Chrono-Amperometry step with a duration (decimation off)."
        )
    if scope["current_conversion_factor_error"]:
        raise ValueError(scope["current_conversion_factor_error"])
    scope_cls = hw.Picoscope5000a if scope["model"] == "Picoscope5000a" else hw.Picoscope4000
    pico = scope_cls(scope["resolution"])
    pico.set_pico(
        scope["capture_size"], scope["capture_size"], scope["sampling_time_value"], scope["sampling_time_unit"],
        str(saving_dir),
    )
    pico.set_channel("PS5000A_CHANNEL_A", scope["range_a"])
    # set_channel's signature is (channel, vrange, signal_name=None, conv_factor=None): conv_factor MUST be
    # passed by keyword -- positionally it would be taken as signal_name and the "current" would be the raw
    # scope voltage on channel B, independent of the I range.
    pico.set_channel("PS5000A_CHANNEL_B", scope["range_b"], conv_factor=scope["current_conversion_factor"])

    online = config["online_analysis"]
    if not online["enabled"]:
        from elma.runs import RawCaptureRun
        return RawCaptureRun(
            channel=channel, pico=pico, awg_channels=awg_channels, saving_dir=saving_dir,
            conditions=conditions, awg=awg,
        )

    from deistools.processing import FermiDiracFilter, MultiFrequencyAnalysis

    from elma.blockcalculator import BlockCalculator
    from elma.deischannel import DEISchannel
    from elma.picocalculator import PicoCalculator

    window_size = online["window_size"]
    sampling_time = scope["sampling_time_seconds"]
    high_z_calculator = MultiFrequencyAnalysis(frequencies, np.zeros(window_size), np.zeros(window_size), sampling_time)
    high_z_calculator.compute_freq_axis()
    lp_filter = FermiDiracFilter(high_z_calculator.freq_axis, 0, 2 * online["filter_cutoff"], online["filter_order"])
    resampling_frequency = float(online["resampling_frequency"])
    ds_factor = int((1 / sampling_time) // resampling_frequency)
    buffer_size = int(float(online["buffer_duration"]) * resampling_frequency)
    block_calculator = BlockCalculator(
        input_size=window_size, sampling_time=sampling_time, high_z_calculator=high_z_calculator,
        lp_filter=lp_filter, ds_factor=ds_factor, buffer_size=buffer_size, potentiostat=channel,
        save_dir=saving_dir,
    )
    pico_calculator = PicoCalculator(
        pico=pico, block_calculator=block_calculator, potentiostat=channel, deis_indexes=deis_indexes,
    )
    return DEISchannel(
        potentiostat=channel, pico=pico_calculator, frequencies=frequencies, awg=awg,
        conditions=conditions, deis_indexes=deis_indexes,
    )
