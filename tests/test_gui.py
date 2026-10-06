"""Smoke/behaviour tests of the GUI tabs, run headless (offscreen Qt). Skipped without PyQt5."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class ExperimentBuilder(unittest.TestCase):
    def setUp(self):
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        self.tab = ExperimentBuilderTab()

    def _add(self, index):
        self.tab.technique_type_combo.setCurrentIndex(index)
        self.tab._on_add_technique_step()

    def test_steps_flags_and_defaults(self):
        for index in (0, 4, 2, 5, 7):          # CA, OCV, CP, PEIS, Loop
            self._add(index)
        flags = [self.tab._sequence_specs[i].get("deis") for i in range(5)]
        self.assertEqual(flags, [True, False, True, False, None])
        self.tab.sequence_list.item(1).setCheckState(Qt.Checked)        # tick the OCV step
        self.assertTrue(self.tab._sequence_specs[1]["deis"])
        self.assertFalse(bool(self.tab.sequence_list.item(4).flags() & Qt.ItemIsUserCheckable))   # Loop: no flag
        self.tab.sequence_list.setCurrentRow(0)
        self.tab._on_remove_technique_step()
        self.assertEqual(len(self.tab._sequence_specs), 4)
        self.assertTrue(self.tab.sequence_list.item(0).text().startswith("1. OCV"))

    def test_settings_round_trip_and_the_old_file_tag(self):
        self._add(0)
        self._add(2)
        saved = self.tab._gather_settings_for_save()
        self.assertEqual(saved["type"], "elma_experiment_settings")
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        other = ExperimentBuilderTab()
        other._apply_loaded_settings(json.loads(json.dumps(saved)))
        self.assertEqual([s["type"] for s in other._sequence_specs], ["CA", "CP"])
        self.assertEqual(other._gather_settings_for_save(), saved)

    def test_configuration_is_the_library_resolution_of_the_settings(self):
        from elma import config as C
        self._add(0)
        self.tab.scope_enabled_check.setChecked(False)
        cfg = self.tab._gather_configuration()
        self.assertEqual(cfg, C.resolve_configuration(self.tab._gather_settings_for_save(), None, None))
        self.assertFalse(cfg["oscilloscope"]["enabled"])
        self.assertEqual(cfg["potentiostat"]["e_range"], "E_RANGE_5V")

    def test_start_hands_the_resolved_configuration_to_the_builder(self):
        import elma.gui.tabs.experiment_builder_tab as module
        seen = {}
        original = module.build_run
        module.build_run = lambda config, ms, split: seen.update(config=config, ms=ms, split=split) or "RUN"
        try:
            self._add(0)
            self.assertEqual(self.tab._build_run(), "RUN")
        finally:
            module.build_run = original
        self.assertEqual(seen["config"]["sequence"][0]["type"], "CA")
        self.assertIsNone(seen["ms"])

    def test_plot_toolbars(self):
        from elma.gui.plot_toolbar import ZoomToolbar
        self.assertIsInstance(self.tab.plots_toolbar, ZoomToolbar)
        self.assertIsInstance(self.tab.peis_geis_toolbar, ZoomToolbar)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class Inspection(unittest.TestCase):
    FS = 10.0
    FREQS = np.array([0.05, 0.1, 0.25, 1.0, 2.0])

    def _make_experiment(self, root):
        n = 20000
        t = np.arange(n) / self.FS
        low = self.FREQS[:3]
        w = 2 * np.pi * low
        z = 1.0 + 3 / (1 + 1j * w * 2.0)
        ph = np.array([0.3, 1.0, 2.0])
        current = -1e-4 + sum(1e-5 * np.cos(2 * np.pi * f * t + p) for f, p in zip(low, ph))
        drift = 1.0 + 1.5 * np.exp(-t / 150.0) + 0.0005 * t
        voltage = drift + sum(1e-5 * abs(zz) * np.cos(2 * np.pi * f * t + p + np.angle(zz)) for f, p, zz in zip(low, ph, z))
        half = n // 2
        for k, sl in ((0, slice(0, half)), (2, slice(half, n))):
            d = root / "pico_aquisition" / f"cycle_0_sequence_{k}"
            d.mkdir(parents=True)
            np.save(d / "voltage.npy", voltage[sl].astype(np.float32))
            np.save(d / "current.npy", current[sl].astype(np.float32))
            np.save(d / "impedance.npy", np.full((5, 10), 5.0 + 0j, dtype=np.complex64))
        (root / "metadata_deis_exp.json").write_text(json.dumps({"Frequencies multisine (Hz)": self.FREQS.tolist()}))
        times = np.arange(0, n / self.FS, 1.0)
        rows = np.column_stack([times, np.interp(times, t, voltage), np.interp(times, t, current),
                                np.zeros(times.size), np.zeros(times.size)])
        np.savetxt(root / "measurement_data.txt", rows, header="Time/s\tEwe/V\tI/A\tTechnique_num\tLoop_num",
                   delimiter="\t", comments="")
        return z, drift, t

    def setUp(self):
        from elma.gui.tabs.inspection_tab import InspectionTab
        self.tab = InspectionTab()
        self._critical = QMessageBox.critical

        def fail(*args, **kwargs):
            raise AssertionError(f"QMessageBox.critical: {args[1:]}")
        QMessageBox.critical = staticmethod(fail)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.z_true, self.drift, self.t = self._make_experiment(self.root)

    def tearDown(self):
        QMessageBox.critical = self._critical
        self.tmp.cleanup()

    def test_load_reads_through_the_deistools_loader(self):
        self.tab._load_from_folder(self.root)
        self.assertEqual(self.tab._decimated_voltage.size, 20000)
        self.assertEqual(self.tab._impedance.shape, (5, 20))
        np.testing.assert_allclose(self.tab._frequencies, self.FREQS)
        self.assertAlmostEqual(float(np.mean(self.tab._current)), -0.1, places=2)     # A -> mA
        self.assertIn("cycle_0_sequence_0, cycle_0_sequence_2", self.tab.load_status_label.text())

    def test_dmfa_with_spline_detrend_recovers_the_impedance_and_fills_the_views(self):
        from deistools.processing import detrending
        self.tab._load_from_folder(self.root)
        self.tab.dmfa_cutoff_spin.setValue(0.5)
        self.tab.dmfa_resampling_spin.setValue(self.FS)
        self.tab.dmfa_detrend_combo.setCurrentText(detrending.METHOD_SPLINE)
        self.tab._on_run_dmfa()
        z = self.tab._impedance
        self.assertEqual(z.shape, (1 + 3 + 2, 100))                                    # DC row, 3 low freqs, 2 online rows
        np.testing.assert_allclose(self.tab._frequencies, [0.0, 0.05, 0.1, 0.25, 1.0, 2.0])
        inner = slice(10, -10)
        err = np.abs(z[1:4, inner] - self.z_true[:, None]) / np.abs(self.z_true)[:, None]
        self.assertLess(np.median(err), 0.01)
        np.testing.assert_allclose(z[4:, :], 5.0)                                      # online rows held, untouched
        self.assertEqual(self.tab.block_slider.maximum(), 99)
        # the DC voltage has the trend put back
        dc = np.interp(self.tab._display_times, self.t, self.drift)
        np.testing.assert_allclose(self.tab._zero_voltage[5:-5], dc[5:-5], atol=3e-3)
        # the detrend / spectrum section shows this very run
        view = self.tab._detrend_view
        self.assertEqual(view["method"], detrending.METHOD_SPLINE)
        np.testing.assert_allclose(view["raw_v"][:100] - view["v_trend"][:100], view["v_det"][:100], atol=1e-6)
        self.assertIn("Smooth spline", self.tab.dmfa_status_label.text())

    def test_detrend_view_can_be_updated_without_running_dmfa(self):
        self.tab._load_from_folder(self.root)
        self.tab.dmfa_cutoff_spin.setValue(0.5)
        self.tab.dmfa_resampling_spin.setValue(self.FS)
        self.tab._on_update_detrend_plot()
        self.assertIsNotNone(self.tab._detrend_view)
        self.assertIn("20,000 samples", self.tab.detrend_status_label.text())

    def test_every_detrend_method_runs(self):
        from deistools.processing import detrending
        self.tab._load_from_folder(self.root)
        self.tab.dmfa_cutoff_spin.setValue(0.5)
        self.tab.dmfa_resampling_spin.setValue(self.FS)
        for method in detrending.METHODS:
            self.tab.dmfa_detrend_combo.setCurrentText(method)
            self.tab._on_run_dmfa()
            self.assertEqual(self.tab._impedance.shape[0], 6)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class MultisineDesigner(unittest.TestCase):
    def test_tab_builds_with_a_zoom_toolbar(self):
        from elma.gui.plot_toolbar import ZoomToolbar
        from elma.gui.tabs.multisine_tab import MultisineDesignerTab
        tab = MultisineDesignerTab()
        self.assertIsInstance(tab.plot_toolbar, ZoomToolbar)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class MainWindowTabs(unittest.TestCase):
    def test_main_window_has_the_three_tabs(self):
        from elma.gui.main_window import MainWindow
        window = MainWindow()
        titles = [window.centralWidget().tabText(i) for i in range(window.centralWidget().count())] \
            if hasattr(window.centralWidget(), "tabText") else None
        self.assertIsNotNone(titles)
        self.assertEqual(len(titles), 3)


if __name__ == "__main__":
    unittest.main()
