"""
A number plus a unit, for the quantities the user types in (potential, current, time).

Same idea as the base-frequency field of the AWG group: a number box next to a unit box. The rest of the GUI
only ever sees the value in SI units (V, A, s): value() returns it, setValue() takes it, ranges are given in it,
and saved settings keep storing it, so settings files are unaffected by which unit was shown.

Changing the unit keeps the number and changes the value (10 with "mA" selected becomes 10 with "uA" selected =
a thousand times smaller), exactly like the base-frequency field; the range of the number box follows the unit.
setValue() (loading a saved value, restoring a step) picks the most natural unit by itself: 0.0001 A shows as
100 uA.
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QSizePolicy, QWidget

# unit label -> value of one unit in SI (the order is the order of the combo box)
VOLTAGE_UNITS = {"V": 1.0, "mV": 1e-3, "\u00b5V": 1e-6}
CURRENT_UNITS = {"A": 1.0, "mA": 1e-3, "\u00b5A": 1e-6, "nA": 1e-9}
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}


def natural_unit(units: dict, value_si: float, default: str) -> str:
    """The largest unit in which |value| is at least 1; `default` for zero (and the smallest unit for tiny values)."""
    if value_si == 0:
        return default
    magnitude = abs(value_si)
    for label, factor in sorted(units.items(), key=lambda item: -item[1]):
        if magnitude >= factor * (1 - 1e-9):
            return label
    return min(units, key=units.get)


def _clean(x: float) -> float:
    """Drops floating-point noise (100 * 1e-6 -> 0.0001 exactly as typed)."""
    return float(f"{x:.12g}")


class QuantityEdit(QWidget):
    """Number box + unit box holding a physical quantity; the value is in SI units."""

    valueChanged = pyqtSignal(float)  # the value in SI units

    def __init__(self, units: dict = None, unit: str = None, parent=None):
        super().__init__(parent)
        self._units = dict(units or VOLTAGE_UNITS)
        self._lo, self._hi = -1e9, 1e9
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(6)
        self._combo = QComboBox()
        self._combo.addItems(list(self._units))
        self._combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if unit is not None:
            self._combo.setCurrentText(unit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin, 1)
        layout.addWidget(self._combo)
        self._spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Number and unit. Changing the unit keeps the number and changes the value.")
        self._apply_range()
        self._spin.valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))
        self._combo.currentTextChanged.connect(self._on_unit_changed)

    # ------------------------------------------------------------------ value (SI)
    def value(self) -> float:
        return _clean(self._spin.value() * self._units[self._combo.currentText()])

    def setValue(self, value_si: float):
        value_si = min(max(float(value_si), self._lo), self._hi)
        unit = natural_unit(self._units, value_si, self._combo.currentText())
        was = self._combo.blockSignals(True)
        self._combo.setCurrentText(unit)
        self._combo.blockSignals(was)
        self._apply_range()
        was = self._spin.blockSignals(True)
        self._spin.setValue(value_si / self._units[unit])
        self._spin.blockSignals(was)
        self.valueChanged.emit(self.value())

    def number(self) -> float:
        """The number shown in the box (not multiplied by the unit)."""
        return self._spin.value()

    def setNumber(self, number: float):
        """Set the number shown, keeping the unit (like typing into the box)."""
        self._spin.setValue(number)

    # ------------------------------------------------------------------ range, precision (range in SI)
    def setRange(self, lo_si: float, hi_si: float):
        self._lo, self._hi = float(lo_si), float(hi_si)
        self._apply_range()

    def minimum(self) -> float:
        return self._lo

    def maximum(self) -> float:
        return self._hi

    def setDecimals(self, decimals: int):
        self._spin.setDecimals(decimals)

    def decimals(self) -> int:
        return self._spin.decimals()

    def setSingleStep(self, step: float):
        """Step of the arrows, in the unit shown."""
        self._spin.setSingleStep(step)

    # ------------------------------------------------------------------ unit
    def unit(self) -> str:
        return self._combo.currentText()

    def units(self) -> dict:
        return dict(self._units)

    def setUnit(self, label: str):
        self._combo.setCurrentText(label)

    def setUnits(self, units: dict, unit: str = None):
        """Replace the available units (e.g. volts <-> amperes when the variable changes). The number stays,
        the unit becomes `unit` or the first (largest) of the new ones."""
        self._units = dict(units)
        was = self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(list(self._units))
        self._combo.setCurrentText(unit if unit in self._units else next(iter(self._units)))
        self._combo.blockSignals(was)
        self._apply_range()
        self.valueChanged.emit(self.value())

    # ------------------------------------------------------------------ internals
    def _apply_range(self):
        factor = self._units[self._combo.currentText()]
        lo, hi = sorted((self._lo / factor, self._hi / factor))
        was = self._spin.blockSignals(True)
        self._spin.setRange(lo, hi)  # clamps the number if needed
        self._spin.blockSignals(was)

    def _on_unit_changed(self, _label):
        self._apply_range()
        self.valueChanged.emit(self.value())
