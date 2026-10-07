"""The number + unit fields of the Experiment Builder (headless Qt; skipped without PyQt5)."""
import json
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False

if HAVE_QT:
    from elma.gui.quantity import CURRENT_UNITS, TIME_UNITS, VOLTAGE_UNITS, QuantityEdit, TimeEdit, natural_unit

UA = "\u00b5A"


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class NaturalUnit(unittest.TestCase):
    def test_picks_the_largest_unit_that_keeps_the_number_at_least_one(self):
        self.assertEqual(natural_unit(CURRENT_UNITS, 1e-4, "A"), UA)
        self.assertEqual(natural_unit(CURRENT_UNITS, 2.5e-3, "A"), "mA")
        self.assertEqual(natural_unit(CURRENT_UNITS, 1.0, "mA"), "A")
        self.assertEqual(natural_unit(VOLTAGE_UNITS, -0.05, "V"), "mV")
        self.assertEqual(natural_unit(CURRENT_UNITS, 3e-12, "A"), "nA")           # smaller than every unit
        self.assertEqual(natural_unit(TIME_UNITS, 7200.0, "s"), "h")

    def test_zero_keeps_the_default(self):
        self.assertEqual(natural_unit(CURRENT_UNITS, 0.0, "mA"), "mA")


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class Widget(unittest.TestCase):
    def test_value_is_si_and_set_value_chooses_the_unit(self):
        w = QuantityEdit(CURRENT_UNITS)
        w.setRange(-10, 10)
        w.setValue(1e-4)
        self.assertEqual((w.unit(), w.number(), w.value()), (UA, 100.0, 1e-4))
        w.setValue(0.05)
        self.assertEqual((w.unit(), w.value()), ("mA", 0.05))
        w.setValue(-2.5)
        self.assertEqual((w.unit(), w.value()), ("A", -2.5))

    def test_changing_the_unit_keeps_the_number(self):
        w = QuantityEdit(CURRENT_UNITS)
        w.setUnit("mA")
        w.setNumber(10)
        self.assertAlmostEqual(w.value(), 0.01)
        w.setUnit(UA)
        self.assertEqual((w.number(), w.value()), (10.0, 1e-5))

    def test_the_range_is_in_si_and_follows_the_unit(self):
        w = QuantityEdit(VOLTAGE_UNITS)
        w.setRange(-10, 10)
        w.setUnit("mV")
        w.setNumber(1e9)
        self.assertEqual(w.value(), 10.0)                      # clamped to the SI range (10000 mV)
        w.setUnit("V")
        self.assertEqual(w.value(), 10.0)
        w.setNumber(-1e9)
        self.assertEqual(w.value(), -10.0)
        w.setValue(99)                                          # out of range: clamped
        self.assertEqual(w.value(), 10.0)

    def test_value_has_no_floating_point_noise(self):
        w = QuantityEdit(CURRENT_UNITS)
        w.setUnit(UA)
        w.setNumber(100)
        self.assertEqual(w.value(), 1e-4)                      # not 9.999999999999999e-05
        w.setUnit("nA")
        w.setNumber(0.1)
        self.assertEqual(w.value(), 1e-10)

    def test_signal_carries_the_si_value(self):
        w = QuantityEdit(VOLTAGE_UNITS)
        seen = []
        w.valueChanged.connect(seen.append)
        w.setUnit("mV")
        w.setNumber(50)
        w.setValue(2.0)
        self.assertEqual(seen, [0.0, 0.05, 2.0])           # unit change (number 0), number typed, value set

    def test_set_units_swaps_volts_for_amperes(self):
        w = QuantityEdit(VOLTAGE_UNITS)
        w.setRange(-1e6, 1e6)
        w.setNumber(0.3)
        w.setUnits(CURRENT_UNITS)
        self.assertEqual((w.unit(), w.number(), w.value()), ("A", 0.3, 0.3))
        self.assertEqual(list(w.units()), ["A", "mA", UA, "nA"])


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class ExperimentBuilderFields(unittest.TestCase):
    def setUp(self):
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        self.tab = ExperimentBuilderTab()

    def _add(self, index):
        self.tab.technique_type_combo.setCurrentIndex(index)
        self.tab._on_add_technique_step()

    def test_ca_voltage_in_millivolts_is_stored_in_volts(self):
        self.tab.ca_voltage_spin.setUnit("mV")
        self.tab.ca_voltage_spin.setNumber(-250)
        self._add(0)
        self.assertEqual(self.tab._sequence_specs[0]["voltage"], -0.25)

    def test_cp_current_in_microamperes_and_record_every_dE(self):
        self.tab.cp_current_spin.setUnit(UA)
        self.tab.cp_current_spin.setNumber(-100)
        self.tab.cp_record_dE_spin.setUnit("mV")
        self.tab.cp_record_dE_spin.setNumber(5)
        self._add(2)
        spec = self.tab._sequence_specs[0]
        self.assertEqual((spec["current"], spec["record_dE"]), (-1e-4, 0.005))

    def test_labels_no_longer_carry_a_unit(self):
        text = " | ".join(w.text() for w in self.tab.findChildren(QLabel))
        for stale in ("Voltage (V)", "Record every dI (mA)", "Record every dE (mV)", "Initial voltage step (V)",
                      "Sine amplitude (V)", "Sine amplitude (A)", "Initial current step (A)"):
            self.assertNotIn(stale, text)

    def test_defaults_show_a_natural_unit(self):
        self.assertEqual((self.tab.peis_amplitude_spin.unit(), self.tab.peis_amplitude_spin.number()), ("mV", 50.0))
        self.assertEqual((self.tab.awg_amplitude_spin.unit(), self.tab.awg_amplitude_spin.number()), ("mV", 50.0))
        self.assertEqual(self.tab.ca_record_dI_spin.unit(), "A")                  # default 1 A (= never record on dI)

    def test_limit_value_follows_the_limit_variable(self):
        spin = self.tab.calim_limit_value_spin
        self.assertEqual(list(spin.units()), list(VOLTAGE_UNITS))
        self.tab.calim_limit_type_combo.setCurrentText("I")
        self.assertEqual(list(spin.units()), list(CURRENT_UNITS))
        spin.setUnit("mA")
        spin.setNumber(2)
        self._add(1)
        self.assertEqual(self.tab._sequence_specs[0]["limit_value"], 0.002)

    def test_awg_amplitude_follows_the_amplitude_type(self):
        spin = self.tab.awg_amplitude_spin
        self.tab.awg_amplitude_type_combo.setCurrentText("Current (A)")
        self.assertEqual(list(spin.units()), list(CURRENT_UNITS))
        spin.setUnit(UA)
        spin.setNumber(100)
        self.assertEqual(self.tab._gather_settings_for_save()["awg"]["amplitude_input"], 1e-4)
        self.tab.awg_amplitude_type_combo.setCurrentText("Potential (V)")
        self.assertEqual(list(spin.units()), list(VOLTAGE_UNITS))

    def test_software_limit_threshold_units_follow_the_quantity(self):
        spin = self.tab.condition_threshold_spin
        self.tab.condition_quantity_combo.setCurrentText("I")
        self.assertEqual(list(spin.units()), list(CURRENT_UNITS))
        spin.setUnit("mA")
        spin.setNumber(0.3)
        self.tab._on_add_condition()
        self.assertAlmostEqual(self.tab._condition_specs[0]["threshold"], 3e-4)
        self.assertIn("A (avg over", self.tab.conditions_list.item(0).text())
        self.tab.condition_quantity_combo.setCurrentText("ElapsedTime")
        self.assertEqual(list(spin.units()), list(TIME_UNITS))
        self.tab.condition_quantity_combo.setCurrentText("Ewe")
        self.assertEqual(list(spin.units()), list(VOLTAGE_UNITS))

    def test_saved_settings_stay_in_si_and_reload_with_a_natural_unit(self):
        self.tab.cp_current_spin.setUnit(UA)
        self.tab.cp_current_spin.setNumber(-100)
        self._add(2)
        self.tab.awg_amplitude_spin.setUnit("mV")
        self.tab.awg_amplitude_spin.setNumber(20)
        saved = self.tab._gather_settings_for_save()
        self.assertEqual(saved["sequence"][0]["current"], -1e-4)
        self.assertEqual(saved["awg"]["amplitude_input"], 0.02)
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        other = ExperimentBuilderTab()
        other._apply_loaded_settings(saved)
        self.assertEqual((other.awg_amplitude_spin.unit(), other.awg_amplitude_spin.number()), ("mV", 20.0))
        self.assertEqual(other._gather_settings_for_save(), saved)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class TimeEditWidget(unittest.TestCase):
    def test_value_in_seconds_is_split_into_h_min_s(self):
        w = TimeEdit()
        w.setValue(3725.5)
        self.assertEqual(w.hms(), (1, 2, 5.5))
        self.assertEqual(w.value(), 3725.5)
        w.setValue(30)
        self.assertEqual((w.hms(), w.value()), ((0, 0, 30.0), 30.0))

    def test_typing_into_the_boxes_sets_the_total(self):
        w = TimeEdit()
        w.setHMS(2, 30, 15.25)
        self.assertEqual(w.value(), 2 * 3600 + 30 * 60 + 15.25)
        seen = []
        w.valueChanged.connect(seen.append)
        w._minutes.setValue(45)
        self.assertEqual(seen[-1], 2 * 3600 + 45 * 60 + 15.25)

    def test_the_arrows_carry_and_borrow(self):
        w = TimeEdit()
        w.setHMS(0, 0, 59)
        w._seconds.stepBy(1)                    # 59 s + 1 s
        self.assertEqual(w.hms(), (0, 1, 0.0))
        w._seconds.stepBy(-1)
        self.assertEqual(w.hms(), (0, 0, 59.0))
        w.setHMS(0, 59, 0)
        w._minutes.stepBy(1)
        self.assertEqual(w.hms(), (1, 0, 0.0))
        w.setValue(0)
        w._seconds.stepBy(-1)                   # cannot go below the minimum
        self.assertEqual(w.value(), 0.0)

    def test_total_is_limited_to_the_range(self):
        w = TimeEdit()
        w.setRange(0, 7200)
        w.setValue(99999)
        self.assertEqual(w.value(), 7200.0)
        self.assertEqual(w.hms(), (2, 0, 0.0))
        w.setHMS(2, 59, 0)
        self.assertEqual(w.value(), 7200.0)     # typed past the end: clamped

    def test_rounding_never_leaves_sixty_seconds(self):
        w = TimeEdit()
        w.setValue(59.9996)
        self.assertEqual(w.hms(), (0, 1, 0.0))

    def test_decimals_of_the_seconds(self):
        w = TimeEdit()
        w.setValue(1.23456)
        self.assertEqual(w.value(), 1.235)
        w.setDecimals(1)
        w.setValue(1.26)
        self.assertEqual(w.value(), 1.3)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class ExperimentBuilderDurations(unittest.TestCase):
    def setUp(self):
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        self.tab = ExperimentBuilderTab()

    def _add(self, index):
        self.tab.technique_type_combo.setCurrentIndex(index)
        self.tab._on_add_technique_step()

    def test_step_durations_are_entered_as_h_min_s_and_stored_in_seconds(self):
        self.tab.ca_duration_spin.setHMS(1, 2, 3)
        self._add(0)
        self.tab.ocv_duration_spin.setHMS(0, 0, 45)
        self._add(4)
        self.assertEqual([s["duration"] for s in self.tab._sequence_specs], [3723.0, 45.0])
        self.assertEqual(self.tab._sequence_specs[0]["type"], "CA")

    def test_default_duration_is_thirty_seconds(self):
        for name in ("ca", "calim", "cp", "cplim", "ocv"):
            self.assertEqual(getattr(self.tab, f"{name}_duration_spin").hms(), (0, 0, 30.0))
        self.assertEqual(self.tab.peis_duration_step_spin.hms(), (0, 1, 0.0))

    def test_peis_estimate_follows_the_hold_time(self):
        self.tab.peis_duration_step_spin.setHMS(0, 5, 0)
        self.assertIn("300 s initial hold", self.tab.peis_duration_estimate_label.text())
        self.tab.geis_duration_step_spin.setHMS(0, 0, 10)
        self.assertIn("10 s initial hold", self.tab.geis_duration_estimate_label.text())

    def test_duration_labels_carry_no_unit(self):
        text = " | ".join(w.text() for w in self.tab.findChildren(QLabel))
        self.assertNotIn("Duration (s)", text)

    def test_long_duration_round_trips_through_settings(self):
        self.tab.cp_duration_spin.setHMS(15, 0, 0)               # 54000 s, as in the long runs
        self._add(2)
        saved = self.tab._gather_settings_for_save()
        self.assertEqual(saved["sequence"][0]["duration"], 54000.0)
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        other = ExperimentBuilderTab()
        other._apply_loaded_settings(saved)
        self.assertEqual(other._gather_settings_for_save(), saved)


@unittest.skipUnless(HAVE_QT, "PyQt5 not available")
class RemovedFields(unittest.TestCase):
    """"Number of steps" (EC-Lab Step_number, always 0: one step per technique) and "Repeat (cycles)" (N_Cycles:
    the Loop step repeats) were removed."""

    def setUp(self):
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        self.tab = ExperimentBuilderTab()

    def test_no_number_of_steps_box_and_no_such_key_in_the_specs(self):
        text = " | ".join(w.text() for w in self.tab.findChildren(QLabel))
        self.assertNotIn("Number of steps", text)
        self.assertNotIn("Repeat (cycles)", text)
        for index in (0, 1, 2, 3):                       # CA, CALim, CP, CPLim
            self.tab.technique_type_combo.setCurrentIndex(index)
            self.tab._on_add_technique_step()
        self.assertEqual(len(self.tab._sequence_specs), 4)
        self.assertTrue(all("nb_steps" not in spec and "repeat" not in spec for spec in self.tab._sequence_specs))

    def test_settings_saved_by_earlier_versions_still_load(self):
        self.tab.technique_type_combo.setCurrentIndex(0)
        self.tab._on_add_technique_step()
        saved = self.tab._gather_settings_for_save()
        old = json.loads(json.dumps(saved))
        old["sequence"][0]["nb_steps"] = 0
        old["sequence"][0]["repeat"] = 0
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        other = ExperimentBuilderTab()
        other._apply_loaded_settings(old)
        self.assertEqual(other._gather_settings_for_save(), saved)

    def test_the_loop_step_keeps_its_own_repeat_count(self):
        self.tab.loop_repeat_spin.setValue(3)
        self.tab.loop_start_spin.setValue(0)
        self.tab.technique_type_combo.setCurrentIndex(7)
        self.tab._on_add_technique_step()
        saved = self.tab._gather_settings_for_save()
        self.assertEqual(saved["sequence"][0], {"type": "Loop", "repeat_N": 3, "loop_start": 0})
        from elma.gui.tabs.experiment_builder_tab import ExperimentBuilderTab
        other = ExperimentBuilderTab()
        other._apply_loaded_settings(saved)
        self.assertEqual(other._gather_settings_for_save(), saved)


if __name__ == "__main__":
    unittest.main()
