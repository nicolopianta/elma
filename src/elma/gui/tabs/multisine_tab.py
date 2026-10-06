"""
Multisine Designer tab.

Works in harmonic (dimensionless) units, not real Hz -- the actual scale
factor mapping harmonic number -> Hz is deferred to the (future) Experiment
Builder tab, since IMD-safety, crest factor, and waveform shape are all
scale-invariant: designing at first-harmonic = 1 and scaling later is
equivalent to (and much cheaper than) re-running generation at a different
absolute frequency.

Front-end over ``multisine.Multisine``. The Multisine class' own plot(),
plot_dft() and plot_phase_full() methods each create their own pyplot
figure and block on plt.show(), so they can't be embedded in a Qt widget
as-is. Rather than patch the upstream package, this tab replicates their
drawing logic directly against an embedded matplotlib Figure. If this
pattern proves useful it's a small, isolated change to propose upstream
later (accept an optional ax/figure argument).
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from multisine import Multisine

from elma.gui.plot_decimation import decimate_min_max
from elma.gui.plot_toolbar import canvas_with_toolbar
from elma.design import (
    amplitudes_from_impedance,
    equal_amplitudes,
    generate_imd_safe_frequencies,
    load_multisine_json,
    load_split_multisine_bands,
    multisine_from_dict,
    optimize_phases_differential_evolution,
    optimize_phases_random_search,
    save_multisine_json,
    save_split_multisine_json,
    snap_top_frequency_to_power_of_ten,
    split_multisine,
)

# Above this many excitation harmonics, differential evolution's search
# budget (kept low for interactivity -- see design/phase_optimization.py)
# struggles to reliably improve on the starting phases; warn and point at
# random search instead.
_DE_WARN_N_FREQ = 15


class MultisineDesignerTab(QWidget):
    """Configure, inspect, split and crest-factor-optimize a multisine waveform."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._multisine: Multisine | None = None
        self._split_result: dict | None = None
        self._impedance_experiment = None
        self._frequencies_experiment = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.addWidget(self._build_controls(), stretch=0)
        root.addWidget(self._build_plots(), stretch=1)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)

        # --- Sampling ---
        sampling_group = QGroupBox("Sampling")
        sampling_layout = QVBoxLayout(sampling_group)

        self.sampling_mode_combo = QComboBox()
        self.sampling_mode_combo.addItems(["Multiple of max harmonic", "Absolute value"])
        sampling_layout.addWidget(self.sampling_mode_combo)

        self.sampling_stack = QStackedWidget()

        multiple_widget = QWidget()
        multiple_form = QFormLayout(multiple_widget)
        self.sampling_multiplier_spin = QSpinBox()
        self.sampling_multiplier_spin.setRange(2, 10000)
        self.sampling_multiplier_spin.setValue(10)
        multiple_form.addRow("Multiplier (× max harmonic)", self.sampling_multiplier_spin)
        self.sampling_stack.addWidget(multiple_widget)

        absolute_widget = QWidget()
        absolute_form = QFormLayout(absolute_widget)
        self.sampling_frequency_edit = QLineEdit("1000000")
        absolute_form.addRow("Sampling frequency", self.sampling_frequency_edit)
        self.sampling_stack.addWidget(absolute_widget)

        self.sampling_mode_combo.currentIndexChanged.connect(self.sampling_stack.setCurrentIndex)
        sampling_layout.addWidget(self.sampling_stack)
        layout.addWidget(sampling_group)

        # --- Harmonics ---
        freq_group = QGroupBox("Excitation harmonics")
        freq_layout = QVBoxLayout(freq_group)

        self.freq_mode_combo = QComboBox()
        self.freq_mode_combo.addItems(["Manual list", "Log sweep", "Auto (IMD-safe)"])
        freq_layout.addWidget(self.freq_mode_combo)

        self.freq_stack = QStackedWidget()

        manual_widget = QWidget()
        manual_form = QFormLayout(manual_widget)
        self.manual_freqs_edit = QLineEdit("1, 10, 100, 1000, 10000")
        manual_form.addRow("Harmonics (comma-sep.)", self.manual_freqs_edit)
        self.freq_stack.addWidget(manual_widget)

        sweep_widget = QWidget()
        sweep_form = QFormLayout(sweep_widget)
        self.sweep_start_edit = QLineEdit("1")
        self.sweep_stop_edit = QLineEdit("10000")
        self.sweep_n_spin = QSpinBox()
        self.sweep_n_spin.setRange(2, 500)
        self.sweep_n_spin.setValue(20)
        sweep_form.addRow("Start harmonic", self.sweep_start_edit)
        sweep_form.addRow("Stop harmonic", self.sweep_stop_edit)
        sweep_form.addRow("Number of points", self.sweep_n_spin)
        self.freq_stack.addWidget(sweep_widget)

        auto_widget = QWidget()
        auto_form = QFormLayout(auto_widget)
        self.nd_spin = QDoubleSpinBox()
        self.nd_spin.setRange(0.5, 8.0)
        self.nd_spin.setSingleStep(0.5)
        self.nd_spin.setValue(4.0)
        self.ppd_spin = QSpinBox()
        self.ppd_spin.setRange(1, 100)
        self.ppd_spin.setValue(10)
        self.order_spin = QSpinBox()
        self.order_spin.setRange(2, 6)
        self.order_spin.setValue(4)
        self.order_spin.setToolTip(
            "Max size of intermodulation products checked for. Higher = safer "
            "but much slower (combinatorial search)."
        )
        self.snap_top_freq_check = QCheckBox("Snap top harmonic to 10^decades")
        self.snap_top_freq_check.setChecked(True)
        self.split_aware_generation_check = QCheckBox("Generate for split at harmonic (below)")
        self.split_aware_generation_check.setToolTip(
            "Every harmonic at or above the 'Split at harmonic' value (in the Split section "
            "below) is generated as an exact integer multiple of it, instead of of just the "
            "base harmonic -- so split_multisine's high band can use the split frequency "
            "itself as an exact FFT bin-alignment reference (far fewer samples than the "
            "fallback, which has to assume no such structure exists). Cross-band "
            "intermodulation safety is unaffected: both sides are still searched together, "
            "in one pass, exactly like the unsplit case."
        )
        auto_form.addRow("Decades", self.nd_spin)
        auto_form.addRow("Points per decade", self.ppd_spin)
        auto_form.addRow("IMD order", self.order_spin)
        auto_form.addRow(self.snap_top_freq_check)
        auto_form.addRow(self.split_aware_generation_check)
        self.freq_stack.addWidget(auto_widget)

        self.freq_mode_combo.currentIndexChanged.connect(self.freq_stack.setCurrentIndex)

        freq_layout.addWidget(self.freq_stack)
        layout.addWidget(freq_group)

        # --- Generate ---
        self.generate_button = QPushButton("Generate multisine")
        self.generate_button.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_button)

        self.crest_factor_label = QLabel("Crest factor: —")
        layout.addWidget(self.crest_factor_label)

        # --- Generated harmonics ---
        freqs_display_group = QGroupBox("Generated harmonics")
        freqs_display_layout = QVBoxLayout(freqs_display_group)
        self.frequencies_display = QPlainTextEdit()
        self.frequencies_display.setReadOnly(True)
        self.frequencies_display.setFont(QFont("Consolas", 9))
        self.frequencies_display.setMaximumHeight(80)
        self.frequencies_display.setPlainText("—")
        freqs_display_layout.addWidget(self.frequencies_display)
        layout.addWidget(freqs_display_group)

        # --- Amplitude ---
        # The overall waveform is always normalized to a +/-1 peak right
        # after generation (see _on_generate), so only the *relative*
        # distribution of per-tone amplitudes matters here, not their
        # absolute values.
        amp_group = QGroupBox("Amplitude (waveform normalized to ±1 peak)")
        amp_layout = QVBoxLayout(amp_group)

        self.amp_mode_combo = QComboBox()
        self.amp_mode_combo.addItems(["Equal (default)", "From impedance spectrum"])
        amp_layout.addWidget(self.amp_mode_combo)

        self.amp_stack = QStackedWidget()

        equal_widget = QWidget()
        equal_layout = QVBoxLayout(equal_widget)
        equal_label = QLabel("All tones get the same amplitude.")
        equal_label.setStyleSheet("color: gray;")
        equal_label.setWordWrap(True)
        equal_layout.addWidget(equal_label)
        self.amp_stack.addWidget(equal_widget)

        impedance_widget = QWidget()
        impedance_layout = QVBoxLayout(impedance_widget)
        self.load_impedance_button = QPushButton("Load impedance (.npy, complex)...")
        self.load_impedance_button.clicked.connect(self._on_load_impedance)
        self.load_impedance_freqs_button = QPushButton("Load its frequency axis (.npy)...")
        self.load_impedance_freqs_button.clicked.connect(self._on_load_impedance_frequencies)
        self.impedance_status_label = QLabel("No spectrum loaded.")
        self.impedance_status_label.setStyleSheet("color: gray;")
        self.impedance_status_label.setWordWrap(True)
        impedance_layout.addWidget(self.load_impedance_button)
        impedance_layout.addWidget(self.load_impedance_freqs_button)
        impedance_layout.addWidget(self.impedance_status_label)
        self.amp_stack.addWidget(impedance_widget)

        self.amp_mode_combo.currentIndexChanged.connect(self.amp_stack.setCurrentIndex)
        amp_layout.addWidget(self.amp_stack)
        layout.addWidget(amp_group)

        # --- Phase optimization ---
        opt_group = QGroupBox("Crest-factor optimization")
        opt_layout = QVBoxLayout(opt_group)

        self.opt_method_combo = QComboBox()
        self.opt_method_combo.addItems(["Random search", "Differential evolution"])
        opt_layout.addWidget(self.opt_method_combo)

        self.opt_stack = QStackedWidget()

        random_widget = QWidget()
        random_form = QFormLayout(random_widget)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100000)
        self.iterations_spin.setValue(200)
        random_form.addRow("Random-phase iterations", self.iterations_spin)
        self.opt_stack.addWidget(random_widget)

        de_widget = QWidget()
        de_form = QFormLayout(de_widget)
        self.de_restarts_spin = QSpinBox()
        self.de_restarts_spin.setRange(1, 50)
        self.de_restarts_spin.setValue(3)
        self.de_restarts_spin.setToolTip(
            "Each restart runs a full differential-evolution search -- keep this low."
        )
        de_form.addRow("DE restarts", self.de_restarts_spin)
        self.opt_stack.addWidget(de_widget)

        self.opt_method_combo.currentIndexChanged.connect(self.opt_stack.setCurrentIndex)
        opt_layout.addWidget(self.opt_stack)

        self.optimize_button = QPushButton("Optimize phases")
        self.optimize_button.clicked.connect(self._on_optimize)
        self.optimize_button.setEnabled(False)
        opt_layout.addWidget(self.optimize_button)
        layout.addWidget(opt_group)

        # --- Split into low/high bands ---
        # Placed after amplitude/optimization since splitting reuses
        # whichever amplitudes and phases the multisine currently has --
        # pick those first, then split.
        split_group = QGroupBox("Split into low/high bands")
        split_form = QFormLayout(split_group)
        self.split_frequency_edit = QLineEdit("100")
        split_form.addRow("Split at harmonic", self.split_frequency_edit)
        self.split_button = QPushButton("Split")
        self.split_button.clicked.connect(self._on_split)
        self.split_button.setEnabled(False)
        split_form.addRow(self.split_button)
        self.split_info_label = QLabel("Not split.")
        self.split_info_label.setStyleSheet("color: gray;")
        self.split_info_label.setWordWrap(True)
        split_form.addRow(self.split_info_label)
        self.save_split_button = QPushButton("Save split design (JSON)")
        self.save_split_button.clicked.connect(self._on_save_split)
        self.save_split_button.setEnabled(False)
        split_form.addRow(self.save_split_button)
        layout.addWidget(split_group)

        # --- Save / Load ---
        save_group = QGroupBox("Save / Load design")
        save_layout = QVBoxLayout(save_group)
        self.save_button = QPushButton("Save design (JSON)")
        self.save_button.clicked.connect(self._on_save)
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Saves sampling frequency, harmonics, amplitudes and phases only.")
        save_layout.addWidget(self.save_button)
        self.load_button = QPushButton("Load design (JSON)")
        self.load_button.clicked.connect(self._on_load)
        save_layout.addWidget(self.load_button)
        layout.addWidget(save_group)

        layout.addStretch(1)
        return panel

    def _build_plots(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.figure = Figure(figsize=(7, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_widget, self.plot_toolbar = canvas_with_toolbar(self.canvas, autoscale_on_home=False)
        layout.addWidget(plot_widget)
        return panel

    # ------------------------------------------------------------------ #
    # Input parsing
    # ------------------------------------------------------------------ #

    def _parse_frequencies(self) -> np.ndarray:
        mode = self.freq_mode_combo.currentIndex()
        if mode == 0:
            text = self.manual_freqs_edit.text()
            values = [float(v.strip()) for v in text.split(",") if v.strip()]
            if len(values) < 2:
                raise ValueError("Provide at least two harmonics.")
            frequencies = np.array(sorted(values), dtype=float)
        elif mode == 1:
            start = float(self.sweep_start_edit.text())
            stop = float(self.sweep_stop_edit.text())
            n = self.sweep_n_spin.value()
            if start <= 0 or stop <= 0 or stop <= start:
                raise ValueError("Log sweep needs 0 < start < stop.")
            frequencies = np.logspace(np.log10(start), np.log10(stop), n)
        else:
            nd = self.nd_spin.value()
            ppd = self.ppd_spin.value()
            order = self.order_spin.value()
            # First harmonic is always fixed at 1 here -- the real-world
            # scale factor is applied later, in the Experiment Builder.
            split_frequency = None
            if self.split_aware_generation_check.isChecked():
                split_frequency = float(self.split_frequency_edit.text())
            frequencies = generate_imd_safe_frequencies(
                nd, ppd, order=order, first_freq=1.0, split_frequency=split_frequency
            )
            if self.snap_top_freq_check.isChecked():
                frequencies = snap_top_frequency_to_power_of_ten(frequencies, nd, first_freq=1.0)
        return frequencies

    def _build_amplitudes(self, frequencies: np.ndarray) -> np.ndarray:
        if self.amp_mode_combo.currentIndex() == 0:
            return equal_amplitudes(frequencies)
        if self._impedance_experiment is None or self._frequencies_experiment is None:
            raise ValueError("Load an impedance spectrum and its frequency axis first.")
        return amplitudes_from_impedance(frequencies, self._impedance_experiment, self._frequencies_experiment)

    # ------------------------------------------------------------------ #
    # Actions -- generation
    # ------------------------------------------------------------------ #

    def _clear_split(self):
        self._split_result = None
        self.save_split_button.setEnabled(False)
        self.split_info_label.setText("Not split.")

    def _on_generate(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                frequencies = self._parse_frequencies()
                if self.sampling_mode_combo.currentIndex() == 0:
                    sampling_frequency = self.sampling_multiplier_spin.value() * frequencies.max()
                else:
                    sampling_frequency = float(self.sampling_frequency_edit.text())
                if sampling_frequency <= 2 * frequencies.max():
                    raise ValueError(
                        "Sampling frequency must be more than twice the highest "
                        "excitation harmonic (Nyquist)."
                    )
                amplitudes = self._build_amplitudes(frequencies)
                self._multisine = Multisine(sampling_frequency, frequencies, amplitudes)
                self._multisine.normalize_waveform()  # default: waveform peak at +/-1
            except Exception as exc:
                QMessageBox.critical(self, "Could not generate multisine", str(exc))
                return
        finally:
            QApplication.restoreOverrideCursor()

        self.optimize_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.split_button.setEnabled(True)
        self._clear_split()
        self._refresh_plots()

    def _on_load_impedance(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load impedance array", "", "NumPy files (*.npy)")
        if not path:
            return
        try:
            self._impedance_experiment = np.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load impedance array", str(exc))
            return
        self._update_impedance_status()

    def _on_load_impedance_frequencies(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load impedance frequency axis", "", "NumPy files (*.npy)")
        if not path:
            return
        try:
            self._frequencies_experiment = np.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load frequency array", str(exc))
            return
        self._update_impedance_status()

    def _update_impedance_status(self):
        z, f = self._impedance_experiment, self._frequencies_experiment
        if z is None and f is None:
            self.impedance_status_label.setText("No spectrum loaded.")
        elif z is None:
            self.impedance_status_label.setText(f"Frequencies loaded ({f.size} pts) -- impedance still needed.")
        elif f is None:
            self.impedance_status_label.setText(f"Impedance loaded ({z.size} pts) -- frequencies still needed.")
        elif z.size != f.size:
            self.impedance_status_label.setText(
                f"MISMATCH: impedance has {z.size} pts, frequencies have {f.size} pts."
            )
        else:
            self.impedance_status_label.setText(
                f"Ready: {z.size} pts, range [{f.min():.4g}, {f.max():.4g}] Hz."
            )

    # ------------------------------------------------------------------ #
    # Actions -- optimization
    # ------------------------------------------------------------------ #

    def _on_optimize(self):
        if self._multisine is None:
            return
        n_freq = self._multisine.frequencies.size
        if self.opt_method_combo.currentIndex() == 1 and n_freq > _DE_WARN_N_FREQ:
            proceed = QMessageBox.question(
                self,
                "Large harmonic set",
                f"This design has {n_freq} tones. Differential evolution's search budget is "
                "kept low for interactivity and often can't reliably improve on the starting "
                "phases at this dimensionality -- Random search scales much better here. "
                "Continue with differential evolution anyway?",
            )
            if proceed != QMessageBox.Yes:
                return

        cf_before = self._multisine.cf
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.opt_method_combo.currentIndex() == 0:
                # optimize_phases_random_search is the same algorithm as
                # Multisine.best_random_phases, parallelized across a
                # thread pool (compute_multisine itself is already fast --
                # see elma.design.waveform -- so this adds
                # multi-core speedup on top).
                phases, cf = optimize_phases_random_search(self._multisine, iterations=self.iterations_spin.value())
                if cf < cf_before:
                    self._multisine.phases = phases
                    self._multisine.waveform = self._multisine.compute_multisine(phases)
                    self._multisine.cf = cf
                else:
                    QMessageBox.information(
                        self,
                        "No improvement found",
                        f"Random search did not beat the current crest factor "
                        f"({cf_before:.4f}; best found was {cf:.4f}). Phases left unchanged.",
                    )
            else:
                phases, cf = optimize_phases_differential_evolution(
                    self._multisine, num_optimizations=self.de_restarts_spin.value()
                )
                if cf < cf_before:
                    self._multisine.phases = phases
                    self._multisine.waveform = self._multisine.compute_multisine(phases)
                    self._multisine.cf = cf
                else:
                    QMessageBox.information(
                        self,
                        "No improvement found",
                        f"Differential evolution did not beat the current crest factor "
                        f"({cf_before:.4f}; best found was {cf:.4f}). Phases left unchanged.",
                    )
        except Exception as exc:
            QMessageBox.critical(self, "Phase optimization failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._clear_split()  # any existing split no longer matches the current phases
        self._refresh_plots()

    # ------------------------------------------------------------------ #
    # Actions -- split
    # ------------------------------------------------------------------ #

    def _on_split(self):
        if self._multisine is None:
            return
        try:
            split_frequency = float(self.split_frequency_edit.text())
            low, high = split_multisine(self._multisine, split_frequency)
        except Exception as exc:
            QMessageBox.critical(self, "Could not split multisine", str(exc))
            return
        self._split_result = {"split_frequency": split_frequency, "low": low, "high": high}
        self.split_info_label.setText(
            f"Low band: {low.frequencies.size} harmonics, fs={low.sampling_frequency:.4g}, cf={low.cf:.4f}\n"
            f"High band: {high.frequencies.size} harmonics, fs={high.sampling_frequency:.4g}, cf={high.cf:.4f}"
        )
        self.save_split_button.setEnabled(True)
        self._refresh_plots()

    def _on_save_split(self):
        if self._split_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save split multisine design", "", "JSON files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_split_multisine_json(
                self._split_result["low"], self._split_result["high"], self._split_result["split_frequency"], path
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not save split design", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Split design saved to:\n{path}")

    # ------------------------------------------------------------------ #
    # Actions -- save / load
    # ------------------------------------------------------------------ #

    def _on_save(self):
        if self._multisine is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save multisine design", "", "JSON files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_multisine_json(self._multisine, path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save design", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Design saved to:\n{path}")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load multisine design", "", "JSON files (*.json)")
        if not path:
            return
        try:
            data = load_multisine_json(path)
            if data.get("type", "single") == "split":
                # load_split_multisine_bands, not multisine_from_dict directly:
                # the high band's number_points is NOT the default fs/f0 (see
                # design/io.py), and older files don't even save it -- this
                # recovers the right value either way.
                low, high = load_split_multisine_bands(data)
                self._multisine = None
                self._split_result = {
                    "split_frequency": float(data["split_frequency"]),
                    "low": low,
                    "high": high,
                }
                self.split_frequency_edit.setText(str(data["split_frequency"]))
                self.split_info_label.setText(
                    f"Loaded split design (split at {self._split_result['split_frequency']:.4g}): "
                    f"low {low.frequencies.size} harmonics, high {high.frequencies.size} harmonics."
                )
            else:
                self._multisine = multisine_from_dict(data)
                self.sampling_frequency_edit.setText(str(self._multisine.sampling_frequency))
                self._clear_split()
        except Exception as exc:
            QMessageBox.critical(self, "Could not load design", str(exc))
            return

        has_single = self._multisine is not None
        self.optimize_button.setEnabled(has_single)
        self.save_button.setEnabled(has_single)
        self.split_button.setEnabled(has_single)
        self.save_split_button.setEnabled(self._split_result is not None)
        self._refresh_plots()

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def _refresh_plots(self):
        self.figure.clear()
        if self._split_result is not None:
            self._draw_split_plots()
        elif self._multisine is not None:
            self._draw_single_plots()
        else:
            self.crest_factor_label.setText("Crest factor: —")
            self.frequencies_display.setPlainText("—")
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw_idle()
        if hasattr(self, "plot_toolbar"):
            self.plot_toolbar.update()  # the figure was rebuilt: drop the old zoom history

    def _draw_single_plots(self):
        ms = self._multisine
        self.crest_factor_label.setText(f"Crest factor: {ms.cf:.4f}")
        int_freqs = np.round(ms.frequencies).astype(int).tolist()
        self.frequencies_display.setPlainText(str(int_freqs))

        ms.fourier_analysis()
        positive = ms.freq_axis > 0

        ax_time = self.figure.add_subplot(2, 1, 1)
        plot_time, plot_waveform = decimate_min_max(ms.time, ms.waveform)
        ax_time.plot(plot_time, plot_waveform)
        ax_time.set_xlabel("Time / s")
        ax_time.set_ylabel("Amplitude")
        ax_time.set_title(self._waveform_title("Waveform", ms.waveform.size, plot_waveform.size))
        ax_time.grid(True)

        ax_dft = self.figure.add_subplot(2, 1, 2)
        ax_dft.vlines(
            ms.frequencies, 0, np.max(np.abs(ms.fft_waveform)), colors="green", label="Ideal harmonics"
        )
        dft_freq, dft_mag = decimate_min_max(ms.freq_axis[positive], np.abs(ms.fft_waveform[positive]))
        ax_dft.plot(dft_freq, dft_mag, "o", markersize=3, label="Multisine")
        ax_dft.set_xscale("log")
        ax_dft.set_yscale("log")
        ax_dft.set_xlabel("Harmonic")
        ax_dft.set_ylabel("Magnitude")
        ax_dft.set_title(self._waveform_title("DFT magnitude", positive.sum(), dft_mag.size))
        ax_dft.legend(fontsize=8)
        ax_dft.grid(True, which="both", alpha=0.3)

    @staticmethod
    def _waveform_title(base_title, n_original, n_displayed):
        if n_displayed >= n_original:
            return base_title
        return f"{base_title} (display decimated: {n_displayed:,} of {n_original:,} pts)"

    def _draw_split_plots(self):
        low = self._split_result["low"]
        high = self._split_result["high"]
        self.crest_factor_label.setText(f"Low cf: {low.cf:.4f}  |  High cf: {high.cf:.4f}")
        low_ints = np.round(low.frequencies).astype(int).tolist()
        high_ints = np.round(high.frequencies).astype(int).tolist()
        self.frequencies_display.setPlainText(f"Low:  {low_ints}\nHigh: {high_ints}")

        low.fourier_analysis()
        high.fourier_analysis()

        ax_low = self.figure.add_subplot(3, 1, 1)
        plot_time, plot_waveform = decimate_min_max(low.time, low.waveform)
        ax_low.plot(plot_time, plot_waveform, color="tab:blue")
        ax_low.set_xlabel("Time / s")
        ax_low.set_ylabel("Amplitude")
        ax_low.set_title(self._waveform_title("Low band waveform", low.waveform.size, plot_waveform.size))
        ax_low.grid(True)

        ax_high = self.figure.add_subplot(3, 1, 2)
        plot_time, plot_waveform = decimate_min_max(high.time, high.waveform)
        ax_high.plot(plot_time, plot_waveform, color="tab:orange")
        ax_high.set_xlabel("Time / s")
        ax_high.set_ylabel("Amplitude")
        ax_high.set_title(self._waveform_title("High band waveform", high.waveform.size, plot_waveform.size))
        ax_high.grid(True)

        ax_dft = self.figure.add_subplot(3, 1, 3)
        for ms, color, label in ((low, "tab:blue", "Low band"), (high, "tab:orange", "High band")):
            positive = ms.freq_axis > 0
            ax_dft.vlines(ms.frequencies, 0, np.max(np.abs(ms.fft_waveform)), colors=color, alpha=0.4)
            dft_freq, dft_mag = decimate_min_max(ms.freq_axis[positive], np.abs(ms.fft_waveform[positive]))
            ax_dft.plot(dft_freq, dft_mag, "o", markersize=3, color=color, label=label)
        ax_dft.set_xscale("log")
        ax_dft.set_yscale("log")
        ax_dft.set_xlabel("Harmonic")
        ax_dft.set_ylabel("Magnitude")
        ax_dft.set_title("DFT magnitude (both bands)")
        ax_dft.legend(fontsize=8)
        ax_dft.grid(True, which="both", alpha=0.3)
