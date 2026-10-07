import unittest

import numpy as np

from elma import config as C


def ca(duration=100.0, deis=True, i_range="100 µA"):
    return {"type": "CA", "voltage": 0.0, "duration": duration, "vs_init": True, "i_range": i_range, "deis": deis}


def cp(duration=100.0, deis=True, i_range="1 mA"):
    return {"type": "CP", "current": 0.0, "duration": duration, "vs_init": True, "i_range": i_range, "deis": deis}


OCV = {"type": "OCV", "duration": 30.0}
LOOP = {"type": "Loop", "repeat_N": 1, "loop_start": 0}


class DeisFlag(unittest.TestCase):
    def test_defaults_by_type(self):
        self.assertEqual([C.spec_deis_enabled(s) for s in (
            {"type": "CA"}, {"type": "CPLim"}, {"type": "OCV"}, {"type": "PEIS"}, {"type": "GEIS"}, LOOP)],
            [True, True, False, False, False, False])

    def test_explicit_flag_wins_except_for_loop(self):
        self.assertTrue(C.spec_deis_enabled({"type": "OCV", "deis": True}))
        self.assertFalse(C.spec_deis_enabled({"type": "CA", "deis": False}))
        self.assertFalse(C.spec_deis_enabled({"type": "Loop", "deis": True}))

    def test_step_indexes(self):
        self.assertEqual(C.deis_step_indexes([ca(), OCV, cp(), cp(deis=False), LOOP]), [0, 2])


class WindowAndCapture(unittest.TestCase):
    def test_window_is_one_period_of_the_lowest_frequency(self):
        window, buffer = C.compute_analysis_window_and_buffer(np.array([0.01, 0.5, 100.0]), [ca(1000.0)], 1e-6)
        self.assertEqual(window, 100_000_000)
        self.assertEqual(buffer, 1000.0)

    def test_buffer_covers_only_flagged_timed_steps(self):
        seq = [ca(100.0), OCV, cp(200.0, deis=False), ca(50.0), LOOP]
        _, buffer = C.compute_analysis_window_and_buffer(np.array([1.0]), seq, 1e-3)
        self.assertEqual(buffer, 150.0)

    def test_buffer_falls_back_to_three_periods(self):
        _, buffer = C.compute_analysis_window_and_buffer(np.array([0.5]), [OCV], 1e-3)
        self.assertEqual(buffer, 6.0)

    def test_nothing_without_a_design(self):
        self.assertEqual(C.compute_analysis_window_and_buffer(None, [ca()], 1e-6), (None, None))

    def test_capture_size_with_decimation_is_three_windows(self):
        self.assertEqual(C.compute_capture_size(np.array([1.0]), [ca()], True, 1e-3), 3000)

    def test_capture_size_without_decimation_is_the_whole_run(self):
        self.assertEqual(C.compute_capture_size(np.array([1.0]), [ca(10.0), OCV], False, 1e-3), 40_000)
        self.assertIsNone(C.compute_capture_size(np.array([1.0]), [LOOP], False, 1e-3))

    def test_capture_size_without_decimation_counts_the_loop_passes(self):
        # OCV 30 s once, then CA 10 s run 3 times (Loop back to step 1, repeat 2 more times)
        seq = [OCV, ca(10.0), loop(2, 1)]
        self.assertEqual(C.compute_capture_size(np.array([1.0]), seq, False, 1e-3), (30 + 3 * 10) * 1000)

    def test_capture_size_with_decimation_ignores_loops(self):
        seq = [ca(10.0), loop(5, 0)]
        self.assertEqual(C.compute_capture_size(np.array([1.0]), seq, True, 1e-3), 3000)


def loop(repeat_n, start):
    return {"type": "Loop", "repeat_N": repeat_n, "loop_start": start}


class PeisGeisRecording(unittest.TestCase):
    def test_zeros_are_replaced_so_the_instrument_does_not_record_every_sample(self):
        self.assertEqual(C.peis_geis_recording({"type": "PEIS", "record_dt": 0.0, "record_dI": 0.0}), (0.1, 1.0, True))
        self.assertEqual(C.peis_geis_recording({"type": "GEIS", "record_dt": 0.0, "record_dE": 0.0}), (0.1, 1.0, True))

    def test_each_value_is_checked_on_its_own(self):
        self.assertEqual(C.peis_geis_recording({"type": "PEIS", "record_dt": 2.0, "record_dI": 0.0}), (2.0, 1.0, True))
        self.assertEqual(C.peis_geis_recording({"type": "GEIS", "record_dt": 0.0, "record_dE": 0.05}), (0.1, 0.05, True))

    def test_positive_values_are_kept(self):
        self.assertEqual(C.peis_geis_recording({"type": "PEIS", "record_dt": 0.5, "record_dI": 1e-3}), (0.5, 1e-3, False))
        self.assertEqual(C.peis_geis_recording({"type": "GEIS", "record_dt": 0.5, "record_dE": 0.1}), (0.5, 0.1, False))

    def test_peis_reads_dI_and_geis_reads_dE(self):
        self.assertEqual(C.peis_geis_recording({"type": "PEIS", "record_dt": 1.0, "record_dI": 0.3, "record_dE": 0.0})[1], 0.3)
        self.assertEqual(C.peis_geis_recording({"type": "GEIS", "record_dt": 1.0, "record_dE": 0.3, "record_dI": 0.0})[1], 0.3)


class StepPasses(unittest.TestCase):
    def test_no_loop_means_one_pass(self):
        self.assertEqual(C.step_passes([ca(), OCV, cp()]), [1, 1, 1])

    def test_loop_repeats_the_steps_from_loop_start(self):
        self.assertEqual(C.step_passes([OCV, ca(), cp(), loop(3, 1)]), [1, 4, 4, 1])
        self.assertEqual(C.step_passes([ca(), cp(), loop(1, 0)]), [2, 2, 1])

    def test_nested_loops_multiply(self):
        seq = [ca(), cp(), loop(2, 1), loop(1, 0)]   # inner: cp x3; outer: ca and the whole inner x2
        self.assertEqual(C.step_passes(seq), [2, 6, 2, 1])   # (the inner Loop step itself runs once per outer pass)

    def test_a_loop_that_does_not_point_back_is_ignored(self):
        self.assertEqual(C.step_passes([ca(), loop(4, 1)]), [1, 1])
        self.assertEqual(C.step_passes([ca(), loop(4, 5)]), [1, 1])
        self.assertEqual(C.step_passes([ca(), loop(0, 0)]), [1, 1])


class Resolvers(unittest.TestCase):
    def test_potential_amplitude_is_used_as_is(self):
        self.assertEqual(C.resolve_awg_amplitude_volts([ca()], "Potential (V)", 0.05), (0.05, None))

    def test_current_amplitude_goes_through_the_i_range(self):
        volts, error = C.resolve_awg_amplitude_volts([ca(), cp(i_range="1 mA")], "Current (A)", 100e-6)
        self.assertIsNone(error)
        self.assertAlmostEqual(volts, 0.1)

    def test_current_amplitude_errors(self):
        self.assertIn("galvanostatic", C.resolve_awg_amplitude_volts([ca()], "Current (A)", 1e-4)[1])
        self.assertIn("needs", C.resolve_awg_amplitude_volts([], "Current (A)", 1e-4)[1])
        self.assertIn("full-scale", C.resolve_awg_amplitude_volts([cp(i_range="Auto")], "Current (A)", 1e-4)[1])
        self.assertIn("exceeds", C.resolve_awg_amplitude_volts([cp(i_range="1 mA")], "Current (A)", 5e-3)[1])

    def test_flagged_galvanostatic_step_is_preferred(self):
        seq = [cp(i_range="10 mA", deis=False), cp(i_range="1 mA", deis=True)]
        self.assertAlmostEqual(C.resolve_awg_amplitude_volts(seq, "Current (A)", 1e-4)[0], 0.1)

    def test_conv_factor_from_the_first_step_with_an_i_range(self):
        # an OCV first (no I range) must not block it; flagged steps are preferred
        seq = [OCV, ca(i_range="10 mA", deis=False), cp(i_range="1 mA")]
        self.assertEqual(C.resolve_scope_conv_factor(seq, 1.0), (1e-3, None))
        self.assertAlmostEqual(C.resolve_scope_conv_factor([cp(i_range="1 mA")], 2.0)[0], 5e-4)

    def test_conv_factor_errors(self):
        self.assertIn("I range", C.resolve_scope_conv_factor([OCV], 1.0)[1])
        self.assertIn("full-scale", C.resolve_scope_conv_factor([cp(i_range="Booster")], 1.0)[1])


SETTINGS = {
    "experiment": {"saving_directory": "D:/x", "experiment_name": "e"},
    "potentiostat": {"address": "USB0", "eclabsdk_path": "C:/sdk/", "channel": 1, "e_range": "5 V",
                     "live_plot": False, "external_control": True},
    "sequence": [ca(1000.0), OCV, cp(50.0, i_range="1 mA")],
    "software_limits": [{"technique_index": 0, "quantity": "Ewe", "operator": "<", "threshold": 0.01, "num_elements": 60}],
    "awg": {"enabled": True, "address": "USB0::X", "channel": 1, "amplitude_type": "Potential (V)",
            "amplitude_input": 0.05, "base_frequency_value": 10.0, "base_frequency_unit": "mHz", "multisine_path": None},
    "oscilloscope": {"enabled": True, "model": "Picoscope5000a", "resolution": "15 bit", "sampling_value": 1,
                     "sampling_unit": "us", "range_a": "±1 V", "range_b": "±500 mV", "conv_factor_vref": 1.0},
    "online_analysis": {"decimation_enabled": True, "filter_cutoff": 10.0, "filter_order": 25, "resampling_frequency": "50"},
}


class _Design:
    frequencies = np.array([0.01, 1.0, 100.0])


class ResolveConfiguration(unittest.TestCase):
    def test_sdk_names_and_derived_values(self):
        cfg = C.resolve_configuration(SETTINGS, multisine=_Design())
        self.assertEqual(cfg["potentiostat"]["e_range"], "E_RANGE_5V")
        self.assertEqual(cfg["oscilloscope"]["resolution"], "PS5000A_DR_15BIT")
        self.assertEqual((cfg["oscilloscope"]["range_a"], cfg["oscilloscope"]["range_b"]), ("PS5000A_1V", "PS5000A_500MV"))
        self.assertEqual(cfg["oscilloscope"]["sampling_time_unit"], "PS5000A_US")
        self.assertAlmostEqual(cfg["oscilloscope"]["sampling_time_seconds"], 1e-6)
        self.assertEqual(cfg["online_analysis"]["window_size"], 100_000_000)
        self.assertEqual(cfg["oscilloscope"]["capture_size"], 300_000_000)
        self.assertEqual(cfg["online_analysis"]["buffer_duration"], 1050.0)      # ca + cp (OCV not flagged)
        self.assertEqual(cfg["oscilloscope"]["current_conversion_factor"], 1e-4)  # first flagged step with a range: ca, 100 µA
        self.assertEqual(cfg["awg"]["amplitude_pp"], 0.05)
        self.assertEqual((cfg["awg"]["multisine_loaded"], cfg["awg"]["multisine_n_harmonics"]), (True, 3))
        self.assertAlmostEqual(cfg["awg"]["base_frequency_hz"], 0.01)
        self.assertEqual(cfg["online_analysis"]["resampling_frequency"], "50")
        self.assertEqual(len(cfg["software_limits"]), 1)

    def test_scope_off_skips_every_scope_value(self):
        settings = {**SETTINGS, "oscilloscope": {**SETTINGS["oscilloscope"], "enabled": False}}
        cfg = C.resolve_configuration(settings, multisine=_Design())
        self.assertFalse(cfg["oscilloscope"]["enabled"])
        self.assertFalse(cfg["online_analysis"]["enabled"])
        self.assertIsNone(cfg["oscilloscope"]["capture_size"])
        self.assertIsNone(cfg["oscilloscope"]["current_conversion_factor"])
        self.assertIsNone(cfg["online_analysis"]["window_size"])

    def test_decimation_off_captures_the_whole_run(self):
        settings = {**SETTINGS, "online_analysis": {**SETTINGS["online_analysis"], "decimation_enabled": False}}
        cfg = C.resolve_configuration(settings, multisine=_Design())
        self.assertFalse(cfg["online_analysis"]["enabled"])
        self.assertEqual(cfg["oscilloscope"]["capture_size"], 1080 * 1_000_000)

    def test_without_a_design_the_dependent_values_are_empty(self):
        cfg = C.resolve_configuration(SETTINGS)
        self.assertFalse(cfg["awg"]["multisine_loaded"])
        self.assertIsNone(cfg["oscilloscope"]["capture_size"])
        self.assertIsNone(cfg["awg"]["base_frequency_hz"])

    def test_old_settings_without_the_deis_key_get_type_defaults(self):
        old = [{k: v for k, v in s.items() if k != "deis"} for s in SETTINGS["sequence"]]
        cfg = C.resolve_configuration({**SETTINGS, "sequence": old}, multisine=_Design())
        self.assertEqual(C.deis_step_indexes(cfg["sequence"]), [0, 2])


class LoadDesign(unittest.TestCase):
    def test_single_and_split_designs_are_rescaled_to_the_base_frequency(self):
        import tempfile
        from pathlib import Path
        from multisine import Multisine
        from elma.design import save_multisine_json, save_split_multisine_json, split_multisine
        freqs = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0])
        ms = Multisine(1000.0, freqs, np.ones(freqs.size), phases=np.linspace(0, 1, freqs.size))
        with tempfile.TemporaryDirectory() as tmp:
            single, split = Path(tmp) / "single.json", Path(tmp) / "split.json"
            save_multisine_json(ms, single)
            low, high = split_multisine(ms, 10.0)
            save_split_multisine_json(low, high, 10.0, split)
            m, s = C.load_design(single, 0.01)
            self.assertIsNone(s)
            np.testing.assert_allclose(m.frequencies, freqs * 0.01)
            m, s = C.load_design(split, 0.01)
            self.assertIsNone(m)
            np.testing.assert_allclose(C.design_frequencies(None, s), freqs * 0.01)
            settings = {"awg": {"multisine_path": str(single), "base_frequency_value": 10.0, "base_frequency_unit": "mHz"}}
            m, s, base = C.load_design_from_settings(settings)
            self.assertAlmostEqual(base, 0.01)
            np.testing.assert_allclose(m.frequencies, freqs * 0.01)
            self.assertEqual(C.load_design_from_settings({"awg": {}})[:2], (None, None))


if __name__ == "__main__":
    unittest.main()
