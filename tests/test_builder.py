import unittest
from types import SimpleNamespace

import numpy as np

from elma import builder, config as C
from elma.deischannel import DEISchannel
from elma.runs import PotentiostatOnlyRun, RawCaptureRun


class Recorder:
    """Stands in for a pyeclab technique / device class: remembers its keyword arguments."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.made = False

    def make_technique(self):
        self.made = True


def _technique_class(name):
    return type(name, (Recorder,), {})


class FakeDevice:
    def __init__(self, address, binary_path=None):
        self.address, self.binary_path = address, binary_path
        self.loaded = None

    def load_sequence(self, channel_num, sequence, display=False):
        self.loaded = sequence


class FakeChannelConfig:
    def __init__(self, live_plot=False, external_control=False):
        self.live_plot, self.external_control = live_plot, external_control


class FakeChannel:
    def __init__(self, device, num, writer=None, config=None):
        self.device, self.num, self.writer, self.config = device, num, writer, config
        self.running = False
        self.function = None
        self.sequence = None

    def load_sequence(self, sequence, ask_ok=False):
        self.sequence = sequence


class FakeAWGDevice:
    instances = []

    def __init__(self, address, channel_num):
        self.address, self.channel_num = address, channel_num
        self.on = None
        self.calls = []
        self.device = SimpleNamespace(write=lambda cmd: None, query=lambda cmd: '+0,"No error"')
        FakeAWGDevice.instances.append(self)

    def clear_ch_mem(self): self.calls.append("clear")
    def load_awf(self, name, data): self.calls.append(("load", name, len(data)))
    def select_awf(self, name): self.calls.append(("select", name))
    def set_Z_out_infinite(self): pass
    def set_sample_rate(self, r): self.calls.append(("rate", r))
    def set_amplitude(self, a): self.calls.append(("amplitude", a))
    def combine_channels(self): self.calls.append("combine")
    def turn_on(self): self.on = True
    def turn_off(self): self.on = False


class FakePico:
    last = None

    def __init__(self, resolution):
        self.resolution = resolution
        self.channels_set = []
        FakePico.last = self

    def set_pico(self, *args):
        self.set_pico_args = args

    def set_channel(self, name, vrange, signal_name=None, conv_factor=None):
        self.channels_set.append((name, vrange, conv_factor))


def _hardware():
    techniques = SimpleNamespace(
        ChronoAmperometry=_technique_class("CA"), ChronoAmperometryWithLimits=_technique_class("CALim"),
        ChronoPotentiometry=_technique_class("CP"), ChronoPotentiometryWithLimits=_technique_class("CPLim"),
        OpenCircuitVoltage=_technique_class("OCV"), Loop=_technique_class("Loop"),
        build_limit=lambda *a, **k: ("limit",) + a,
        generate_xctr_param=lambda cfg: 8 if cfg.external_control else 0,
    )
    return builder.Hardware(
        BiologicDevice=FakeDevice, ChannelConfig=FakeChannelConfig,
        FileWriter=lambda file_dir, experiment_name: SimpleNamespace(file_dir=file_dir, experiment_name=experiment_name),
        Channel=FakeChannel,
        E_RANGE={"E_RANGE_5V": "e5"}, I_RANGE={"I_RANGE_1mA": "i1mA", "I_RANGE_100uA": "i100uA"},
        BANDWIDTH={"BW_9": "bw9"}, techniques=techniques,
        PEISTechnique=_technique_class("PEIS"), GEISTechnique=_technique_class("GEIS"),
        Picoscope4000=FakePico, Picoscope5000a=FakePico, TrueFormAWG=FakeAWGDevice,
    )


def _spec(kind, **kw):
    base = {"CA": dict(voltage=0.0, vs_init=True, nb_steps=0, record_dt=1.0, record_dI=1.0, repeat=0,
                       i_range="100 µA", bandwidth="9"),
            "CP": dict(current=0.0, vs_init=True, nb_steps=0, record_dt=1.0, record_dE=1.0, repeat=0,
                       i_range="1 mA", bandwidth="9"),
            "OCV": dict(record_dt=1.0, bandwidth="9"),
            "PEIS": dict(vs_init=True, voltage_step=0.0, amplitude=0.01, initial_frequency=1e5, final_frequency=1.0,
                         sweep_linear=False, frequency_number=5, average_n_times=3, wait_for_steady=0.1,
                         correction=False, record_dt=0.0, record_dI=0.0, external_control=False),
            "GEIS": dict(vs_init=True, current_step=0.0, amplitude=1e-5, initial_frequency=1e3, final_frequency=10.0,
                         sweep_linear=False, frequency_number=4, average_n_times=3, wait_for_steady=0.1,
                         correction=False, record_dt=0.0, record_dE=0.0, i_range="1 mA", external_control=False),
            "Loop": dict(repeat_N=1, loop_start=0)}[kind]
    return {"type": kind, "duration": 100.0, **base, **kw} if kind != "Loop" else {"type": "Loop", **base}


class _Design:
    frequencies = np.array([1.0, 3.0])
    sampling_frequency = 100.0
    waveform = np.sin(np.linspace(0, 6, 200))


class _Split:
    low = SimpleNamespace(frequencies=np.array([1.0, 3.0]), sampling_frequency=100.0, waveform=np.sin(np.linspace(0, 6, 200)))
    high = SimpleNamespace(frequencies=np.array([30.0, 50.0]), sampling_frequency=1e4, waveform=np.sin(np.linspace(0, 60, 500)))


def _settings(sequence, **overrides):
    s = {
        "experiment": {"saving_directory": "X:/data", "experiment_name": "exp"},
        "potentiostat": {"address": "USB0", "eclabsdk_path": "C:/sdk/", "channel": 1, "e_range": "5 V",
                         "live_plot": False, "external_control": True},
        "sequence": sequence, "software_limits": [],
        "awg": {"enabled": True, "address": "USB0::AWG", "channel": 1, "amplitude_type": "Potential (V)",
                "amplitude_input": 0.05, "base_frequency_value": 1.0, "base_frequency_unit": "Hz"},
        "oscilloscope": {"enabled": True, "model": "Picoscope5000a", "resolution": "15 bit", "sampling_value": 1,
                         "sampling_unit": "ms", "range_a": "±1 V", "range_b": "±500 mV", "conv_factor_vref": 1.0},
        "online_analysis": {"decimation_enabled": True, "filter_cutoff": 10.0, "filter_order": 8, "resampling_frequency": "50"},
    }
    for key, value in overrides.items():
        s[key] = {**s[key], **value} if isinstance(value, dict) else value
    return s


class Sequence(unittest.TestCase):
    def test_every_technique_type_is_built_with_the_right_arguments(self):
        hw = _hardware()
        settings = _settings([_spec("CA"), _spec("OCV"), _spec("CP"), _spec("PEIS"), _spec("GEIS"), _spec("Loop")])
        config = C.resolve_configuration(settings, _Design())
        device = FakeDevice("USB0")
        seq = builder.build_sequence(config, device, hw, FakeChannelConfig(external_control=True))
        self.assertEqual([type(s).__name__ for s in seq], ["CA", "OCV", "CP", "PEIS", "GEIS", "Loop"])
        self.assertTrue(all(s.made for s in seq))
        ca, ocv, cp, peis, geis, loop = seq
        self.assertEqual((ca.kwargs["e_range"], ca.kwargs["i_range"], ca.kwargs["bandwidth"], ca.kwargs["xctr"]),
                         ("e5", "i100uA", "bw9", 8))
        self.assertEqual(cp.kwargs["i_range"], "i1mA")
        self.assertEqual(ocv.kwargs["xctr"], 8)
        # PEIS/GEIS have their own external-control setting: off although the sequence-wide one is on
        self.assertEqual((peis.kwargs["xctr"], geis.kwargs["xctr"]), (0, 0))
        self.assertEqual(geis.kwargs["i_range"], "i1mA")
        self.assertEqual(peis.kwargs["initial_voltage_step"], 0.0)
        self.assertEqual((loop.kwargs["repeat_N"], loop.kwargs["loop_start"]), (1, 0))

    def test_peis_step_can_enable_its_own_external_control(self):
        hw = _hardware()
        config = C.resolve_configuration(_settings([_spec("PEIS", external_control=True)]), _Design())
        peis, = builder.build_sequence(config, FakeDevice("U"), hw, FakeChannelConfig(external_control=False))
        self.assertEqual(peis.kwargs["xctr"], 8)

    def test_unknown_step_type_is_refused(self):
        hw = _hardware()
        config = C.resolve_configuration(_settings([{"type": "Foo", "duration": 1.0}]), _Design())
        with self.assertRaises(ValueError):
            builder.build_sequence(config, FakeDevice("U"), hw, FakeChannelConfig())


class BuildRun(unittest.TestCase):
    def setUp(self):
        FakeAWGDevice.instances.clear()

    def test_deis_run_follows_the_flags(self):
        seq = [_spec("CA"), _spec("OCV"), _spec("CP"), _spec("CP", deis=False), _spec("CA")]
        config = C.resolve_configuration(_settings(seq), _Design())
        run = builder.build_run(config, _Design(), None, hardware=_hardware())
        self.assertIsInstance(run, DEISchannel)
        self.assertEqual(run.deis_indexes, [0, 2, 4])
        self.assertEqual(run.pico.deis_indexes, [0, 2, 4])
        self.assertEqual(run.awg.sequence_indexes, [0, 2, 4])
        self.assertEqual(len(run.awg.amplitudes), 3)
        self.assertEqual(run.awg.amplitudes[0], 0.05)
        self.assertEqual(run.pico.block_calculator.input_size, 1000)                 # 1 period of 1 Hz at 1 ms
        self.assertEqual(str(run.pico.block_calculator.save_dir).replace("\\", "/"), "X:/data/exp")
        self.assertEqual(run.pico.pico.set_pico_args[:2], (3000, 3000))
        self.assertIn(("PS5000A_CHANNEL_B", "PS5000A_500MV", 1e-4), run.pico.pico.channels_set)  # conv factor by keyword
        self.assertEqual(run.potentiostat.function.__name__, "_execute_on_technique_termination")
        device, = FakeAWGDevice.instances
        self.assertIn(("rate", 100.0), device.calls)                                  # applied at upload, not only on update()
        self.assertIn(("amplitude", 0.05), device.calls)

    def test_split_design_uses_two_channels_combined(self):
        config = C.resolve_configuration(_settings([_spec("CA")]), None, {"low": _Split.low, "high": _Split.high})
        run = builder.build_run(config, None, {"low": _Split.low, "high": _Split.high}, hardware=_hardware())
        low, high = FakeAWGDevice.instances
        self.assertEqual((low.channel_num, high.channel_num), (1, 2))
        self.assertIn("combine", low.calls)
        self.assertEqual(run.awg.channel1.sample_rates, [100.0])
        self.assertEqual(run.awg.channel2.sample_rates, [1e4])
        np.testing.assert_allclose(run.frequencies, [1.0, 3.0, 30.0, 50.0])

    def test_scope_off_gives_a_potentiostat_only_run(self):
        settings = _settings([_spec("CA")], oscilloscope={"enabled": False})
        run = builder.build_run(C.resolve_configuration(settings, _Design()), _Design(), None, hardware=_hardware())
        self.assertIsInstance(run, PotentiostatOnlyRun)
        self.assertIsNotNone(run._awg_follower)

    def test_online_analysis_off_gives_a_raw_capture_run(self):
        settings = _settings([_spec("CA")], online_analysis={"decimation_enabled": False})
        run = builder.build_run(C.resolve_configuration(settings, _Design()), _Design(), None, hardware=_hardware())
        self.assertIsInstance(run, RawCaptureRun)
        self.assertEqual(str(run.saving_dir).replace("\\", "/"), "X:/data/exp")

    def test_inconsistent_configurations_are_refused(self):
        hw = _hardware()
        with self.assertRaisesRegex(ValueError, "no multisine"):
            builder.build_run(C.resolve_configuration(_settings([_spec("CA")]), None), None, None, hardware=hw)
        settings = _settings([_spec("CA")], awg={"amplitude_type": "Current (A)"})
        with self.assertRaisesRegex(ValueError, "galvanostatic"):
            builder.build_run(C.resolve_configuration(settings, _Design()), _Design(), None, hardware=hw)
        settings = _settings([_spec("OCV")], awg={"enabled": False})
        with self.assertRaisesRegex(ValueError, "I range"):
            builder.build_run(C.resolve_configuration(settings, _Design()), _Design(), None, hardware=hw)

    def test_software_limits_become_conditions(self):
        settings = _settings([_spec("CA")], oscilloscope={"enabled": False})
        settings["software_limits"] = [{"technique_index": 0, "quantity": "Ewe", "operator": "<", "threshold": 0.01, "num_elements": 60}]
        run = builder.build_run(C.resolve_configuration(settings, _Design()), _Design(), None, hardware=_hardware())
        self.assertEqual((run.conditions[0].quantity, run.conditions[0].threshold, run.conditions[0].num_elements), ("Ewe", 0.01, 60))


if __name__ == "__main__":
    unittest.main()
