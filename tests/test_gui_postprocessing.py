"""The post-processing (smoothing) section of the Inspection tab and the one-frequency plot (headless Qt)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False

if HAVE_QT:
    from deistools.processing import smoothing
    from elma.gui.tabs.inspection_tab import InspectionTab

FREQUENCIES = [1.0, 3.0, 10.0]
LEVELS = (100.0 - 20j, 300.0 - 80j)          # the impedance of the two techniques (a step between them)
BLOCKS = (24, 18)


def make_folder(root: Path, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    (root / "metadata_deis_exp.json").write_text(json.dumps({"Frequencies multisine (Hz)": FREQUENCIES}))
    n = sum(BLOCKS)
    time = np.arange(n, dtype=float)
    (root / "measurement_data.txt").write_text(
        "Time/s\tEwe/V\tI/A\n" + "\n".join(f"{t}\t{1.0}\t{1e-4}" for t in time))
    for sequence, (blocks, level) in enumerate(zip(BLOCKS, LEVELS)):
        d = root / "pico_aquisition" / f"cycle_0_sequence_{sequence}"
        d.mkdir(parents=True)
        z = level * (1 + np.arange(1, 4)[:, None] * 0.1) * np.ones((3, blocks), dtype=complex)
        z = z + noise * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))
        np.save(d / "impedance.npy", z.astype(np.complex64))
        np.save(d / "voltage.npy", np.full(5 * blocks, 1.0, dtype=np.float32))
        np.save(d / "current.npy", np.full(5 * blocks, 1e-4, dtype=np.float32))


def truth():
    return np.hstack([level * (1 + np.arange(1, 4)[:, None] * 0.1) * np.ones((3, blocks), dtype=complex)
                      for blocks, level in zip(BLOCKS, LEVELS)])


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class PostProcessing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_folder(self.root, noise=2.0)
        self.tab = InspectionTab()
        self.tab._load_from_folder(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def choose(self, method, **spin):
        self.tab.post_method_combo.setCurrentText(method)
        for name, value in spin.items():
            getattr(self.tab, f"post_{name}").setValue(value)

    def test_raw_values_are_shown_until_a_method_is_applied(self):
        self.assertEqual(self.tab.post_method_combo.currentText(), smoothing.METHOD_NONE)
        self.assertIs(self.tab._impedance, self.tab._raw_impedance)
        self.assertFalse(self.tab._postprocessed)
        self.assertIn("none", self.tab.post_status_label.text())

    def test_the_pieces_of_the_run_are_known(self):
        np.testing.assert_array_equal(self.tab._segment_edges, [BLOCKS[0]])

    def test_savgol_smooths_and_leaves_the_raw_array_alone(self):
        raw = self.tab._raw_impedance.copy()
        self.choose(smoothing.METHOD_SAVGOL, window_spin=9, polyorder_spin=2)
        self.tab._on_apply_postprocessing()
        self.assertTrue(self.tab._postprocessed)
        np.testing.assert_array_equal(self.tab._raw_impedance, raw)
        self.assertEqual(self.tab._impedance.shape, raw.shape)
        reference = truth()
        self.assertLess(np.std(self.tab._impedance - reference), 0.65 * np.std(raw - reference))
        self.assertIn("Savitzky-Golay", self.tab.post_status_label.text())

    def test_applying_twice_does_not_compound(self):
        self.choose(smoothing.METHOD_GAUSSIAN, sigma_spin=3.0)
        self.tab._on_apply_postprocessing()
        once = self.tab._impedance.copy()
        self.tab._on_apply_postprocessing()
        np.testing.assert_array_equal(self.tab._impedance, once)

    def test_reset_brings_the_raw_values_back(self):
        self.choose(smoothing.METHOD_MEDIAN, window_spin=7)
        self.tab._on_apply_postprocessing()
        self.assertFalse(np.array_equal(self.tab._impedance, self.tab._raw_impedance))
        self.tab._on_reset_postprocessing()
        self.assertIs(self.tab._impedance, self.tab._raw_impedance)
        self.assertFalse(self.tab._postprocessed)
        self.assertEqual(self.tab.post_method_combo.currentText(), smoothing.METHOD_NONE)

    def test_a_step_between_two_techniques_stays_sharp_unless_asked_otherwise(self):
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_folder(self.root, noise=0.0)
        tab = InspectionTab()
        tab._load_from_folder(self.root)
        tab.post_method_combo.setCurrentText(smoothing.METHOD_GAUSSIAN)
        tab.post_sigma_spin.setValue(3.0)
        tab._on_apply_postprocessing()
        np.testing.assert_allclose(tab._impedance, truth(), atol=1e-3)         # pieces smoothed apart: unchanged
        tab.post_steps_check.setChecked(False)
        tab._on_apply_postprocessing()
        boundary = BLOCKS[0]
        self.assertGreater(abs(tab._impedance[0, boundary - 1] - truth()[0, boundary - 1]), 5.0)   # blurred over the step

    def test_hampel_removes_a_spike(self):
        self.tab._raw_impedance[1, 10] += 500.0
        self.tab.post_method_combo.setCurrentText(smoothing.METHOD_NONE)
        self.tab.post_hampel_check.setChecked(True)
        self.tab._on_apply_postprocessing()
        self.assertTrue(self.tab._postprocessed)
        self.assertLess(abs(self.tab._impedance[1, 10] - truth()[1, 10]), 20.0)
        self.assertIn("outlier", self.tab.post_status_label.text())

    def test_polar_representation(self):
        self.choose(smoothing.METHOD_SAVGOL, window_spin=9)
        self.tab.post_representation_combo.setCurrentIndex(1)
        self.tab._on_apply_postprocessing()
        reference = truth()
        self.assertLess(np.std(self.tab._impedance - reference), 0.65 * np.std(self.tab._raw_impedance - reference))

    def test_invalid_settings_are_reported_and_leave_the_raw_values(self):
        self.choose(smoothing.METHOD_SAVGOL, window_spin=3, polyorder_spin=5)
        self.tab._on_apply_postprocessing()
        self.assertIs(self.tab._impedance, self.tab._raw_impedance)
        self.assertIn("not applied", self.tab.post_status_label.text())

    def test_method_changes_enable_the_relevant_fields_and_the_usual_ends(self):
        self.tab.post_method_combo.setCurrentText(smoothing.METHOD_GAUSSIAN)
        self.assertTrue(self.tab.post_sigma_spin.isEnabled())
        self.assertFalse(self.tab.post_window_spin.isEnabled())
        self.assertEqual(self.tab.post_edge_combo.currentData(), smoothing.EDGE_NEAREST)
        self.tab.post_method_combo.setCurrentText(smoothing.METHOD_SAVGOL)
        self.assertTrue(self.tab.post_polyorder_spin.isEnabled())
        self.assertEqual(self.tab.post_edge_combo.currentData(), smoothing.EDGE_INTERP)
        self.tab.post_method_combo.setCurrentText(smoothing.METHOD_NONE)
        self.assertFalse(self.tab.post_edge_combo.isEnabled())

    def test_the_window_is_shown_in_seconds(self):
        self.tab.post_window_spin.setValue(11)
        self.assertIn("about", self.tab.post_window_label.text())
        self.assertIn("11 points", self.tab.post_window_label.text())

    def test_the_nyquist_keeps_the_raw_points_in_grey(self):
        self.assertEqual(len(self.tab._nyquist_raw_line.get_xdata()), 0)
        self.choose(smoothing.METHOD_GAUSSIAN, sigma_spin=3.0)
        self.tab._on_apply_postprocessing()
        self.assertEqual(len(self.tab._nyquist_raw_line.get_xdata()), 3)
        self.assertEqual(len(self.tab._nyquist_line.get_xdata()), 3)

    def test_edges_from_counts(self):
        edges = InspectionTab._edges_from_counts([5, 0, 7], 12)
        np.testing.assert_array_equal(edges, [5, 5])
        self.assertEqual(InspectionTab._edges_from_counts([5, 7], 13).size, 0)      # do not add up: unknown
        self.assertEqual(InspectionTab._edges_from_counts([12], 12).size, 0)
        self.assertEqual(InspectionTab._edges_from_counts(None, 12).size, 0)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class OneFrequencyPlot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_folder(self.root, noise=2.0)
        self.tab = InspectionTab()
        self.tab._load_from_folder(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_frequencies_are_listed_and_the_raw_series_drawn(self):
        combo = self.tab.series_frequency_combo
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], FREQUENCIES)
        n = sum(BLOCKS)
        self.assertEqual(len(self.tab._series_raw_line.get_xdata()), n)
        self.assertEqual(len(self.tab._series_line.get_xdata()), 0)         # nothing processed yet

    def test_raw_processed_and_residual_after_smoothing(self):
        self.tab.post_method_combo.setCurrentText(smoothing.METHOD_SAVGOL)
        self.tab._on_apply_postprocessing()
        raw = self.tab._series_raw_line.get_ydata()
        processed = self.tab._series_line.get_ydata()
        residual = self.tab._series_residual_line.get_ydata()
        self.assertEqual(len(processed), sum(BLOCKS))
        np.testing.assert_allclose(residual, raw - processed)
        self.assertLess(np.std(np.diff(processed)), np.std(np.diff(raw)))

    def test_frequency_and_quantity_selection(self):
        self.tab.series_frequency_combo.setCurrentIndex(2)
        self.tab.series_quantity_combo.setCurrentText("Phase / deg")
        y = self.tab._series_raw_line.get_ydata()
        expected = np.degrees(np.unwrap(np.angle(self.tab._raw_impedance[2])))
        np.testing.assert_allclose(y, expected)
        self.tab.series_quantity_combo.setCurrentText("-Z'' / Ohm")
        np.testing.assert_allclose(self.tab._series_raw_line.get_ydata(), -self.tab._raw_impedance[2].imag)

    def test_the_followed_frequency_survives_a_reload(self):
        self.tab.series_frequency_combo.setCurrentIndex(1)
        self.tab._load_from_folder(self.root)
        self.assertEqual(self.tab.series_frequency_combo.currentData(), 3.0)

    def test_nothing_loaded_does_not_fail(self):
        empty = InspectionTab()
        empty._refresh_series_plot()
        empty._on_apply_postprocessing()
        self.assertEqual(empty.series_frequency_combo.count(), 0)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class SavingTheResult(unittest.TestCase):
    def test_the_arrays_and_the_settings_are_written_to_post_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_folder(root, noise=1.0)
            tab = InspectionTab()
            tab._load_from_folder(root)
            tab.post_method_combo.setCurrentText(smoothing.METHOD_SAVGOL)
            tab._on_apply_postprocessing()
            tab._on_save_postprocessed()
            folder = root / "post_processed"
            raw = np.load(folder / "impedance_raw.npy")
            processed = np.load(folder / "impedance_processed.npy")
            self.assertEqual(raw.shape, processed.shape)
            self.assertFalse(np.array_equal(raw, processed))
            np.testing.assert_array_equal(np.load(folder / "frequencies.npy"), FREQUENCIES)
            self.assertEqual(np.load(folder / "times.npy").size, sum(BLOCKS))
            info = json.loads((folder / "postprocessing.json").read_text())
            self.assertTrue(info["postprocessed"])
            self.assertEqual(info["settings"]["method"], smoothing.METHOD_SAVGOL)
            self.assertEqual(info["piece_starts"], [BLOCKS[0]])
            self.assertIn("Saved to", tab.post_status_label.text())


if __name__ == "__main__":
    unittest.main()
