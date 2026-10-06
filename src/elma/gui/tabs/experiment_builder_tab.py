"""
Experiment Builder tab.

Collects everything needed to run a DEIS experiment (potentiostat + technique sequence, AWG,
oscilloscope, online decimation/analysis) and runs it on "Start Experiment". The tab holds only
widgets and plots: the experiment logic lives in the library --
  elma.config   settings -> resolved configuration (capture size, AWG amplitude, scope scales, ...)
  elma.builder  configuration -> run (DEISchannel / RawCaptureRun / PotentiostatOnlyRun)
  elma.runs, elma.peis_geis   run kinds and the PEIS/GEIS channel
so a measurement saved from here can be run without the GUI (see elma.builder).

A split (low/high band) multisine design drives two physical AWG channels, digitally combined into
channel 1's output (one cable to the cell either way). The Multisine Designer's "Auto (IMD-safe)" mode
designs at first harmonic = 1 (dimensionless); the "Base frequency" control in the AWG group rescales a
loaded design (single or split) to a real Hz value -- see elma.design.scaling.

"Preview configuration" only assembles and resolves the values; "Start Experiment" is where the real
objects get built, and fails with a message (not a crash) when no hardware is attached.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from multisine import Multisine

from elma import config as C
from elma.builder import build_run
from elma.design import (
    load_multisine_json,
    load_split_multisine_bands,
    multisine_from_dict,
    rescale_multisine_frequency,
)
from elma.gui.help_button import HelpButton, help_row, with_help
from elma.gui.plot_toolbar import canvas_with_toolbar
from elma.peis_geis import write_peis_geis_csv
from elma.runs import has_finished

# Option lists and label <-> SDK maps shared with the library (elma.config); the underscore names are
# the ones the widget code below has always used.
_RESOLUTION_OPTIONS = C.RESOLUTION_OPTIONS
_TIME_UNIT_OPTIONS = C.TIME_UNIT_OPTIONS
_TIME_UNIT_TO_PICOSDK = C.TIME_UNIT_TO_PICOSDK
_TIME_UNIT_TO_SECONDS = C.TIME_UNIT_TO_SECONDS
_FREQ_UNIT_OPTIONS = C.FREQ_UNIT_OPTIONS
_FREQ_UNIT_TO_HZ = C.FREQ_UNIT_TO_HZ
_VRANGE_OPTIONS = C.VRANGE_OPTIONS
_VRANGE_TO_PICOSDK = C.VRANGE_TO_PICOSDK
_E_RANGE_DISPLAY_TO_SDK = C.E_RANGE_DISPLAY_TO_SDK
_E_RANGE_OPTIONS = C.E_RANGE_OPTIONS
_I_RANGE_DISPLAY_TO_SDK = C.I_RANGE_DISPLAY_TO_SDK
_I_RANGE_OPTIONS = C.I_RANGE_OPTIONS
_I_RANGE_FULL_SCALE_AMPS = C.I_RANGE_FULL_SCALE_AMPS
_BANDWIDTH_DISPLAY_TO_SDK = C.BANDWIDTH_DISPLAY_TO_SDK
_BANDWIDTH_OPTIONS = C.BANDWIDTH_OPTIONS
_LIMIT_TYPE_OPTIONS = C.LIMIT_TYPE_OPTIONS
_LIMIT_SIGN_OPTIONS = C.LIMIT_SIGN_OPTIONS
_LIMIT_LOGIC_OPTIONS = C.LIMIT_LOGIC_OPTIONS
_CONDITION_QUANTITY_OPTIONS = C.CONDITION_QUANTITY_OPTIONS
_DURATION_STEP_TYPES = C.DURATION_STEP_TYPES
_DEIS_OFF_BY_DEFAULT_TYPES = C.DEIS_OFF_BY_DEFAULT_TYPES
_spec_deis_enabled = C.spec_deis_enabled
_resolution_to_picosdk = C.resolution_to_picosdk


class ExperimentBuilderTab(QWidget):
    """Configure and run a DEIS experiment: potentiostat sequence + AWG + scope + online analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sequence_specs = []
        self._condition_specs = []
        self._loaded_multisine: Multisine | None = None
        self._loaded_multisine_split: dict | None = None  # {"low": Multisine, "high": Multisine}
        self._loaded_multisine_data: dict | None = None  # raw parsed JSON, re-scaled on base-frequency change
        self._loaded_multisine_path: str | None = None
        self._deischannel = None
        self._run_status_timer = QTimer(self)
        self._run_status_timer.setInterval(1000)
        self._run_status_timer.timeout.connect(self._on_check_run_finished)
        # Live-plot state: how many rows of measurement_data.txt have been
        # read so far (mirrors pyeclab.LivePlot's own incremental-read
        # approach), the accumulated time/Ewe/I history for the current
        # run, and that run's measurement file path.
        self._plot_lines_read = 0
        self._plot_time = np.array([])
        self._plot_potential = np.array([])
        self._plot_current = np.array([])
        self._plot_measurement_file = None
        # Persists path fields (saving directory, EC-Lab SDK path, last
        # multisine-load folder) across sessions -- an INI file under the
        # user's per-user config dir, not the Windows registry, so it's
        # easy to find/inspect/copy.
        self._settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "elma", "ExperimentBuilder")
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(self._build_experiment_group())
        left.addWidget(self._build_potentiostat_group())
        left.addWidget(self._build_sequence_group())
        left.addWidget(self._build_conditions_group())
        left.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(480)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setMaximumWidth(500)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right = QVBoxLayout()
        right.addWidget(self._build_awg_group())
        right.addWidget(self._build_scope_group())
        right.addWidget(self._build_analysis_group())
        right.addWidget(self._build_run_group())
        right.addStretch(1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMaximumWidth(480)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        right_scroll.setMaximumWidth(500)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        root.addWidget(left_scroll)
        root.addWidget(right_scroll)

        self.log_plots_tabs = QTabWidget()
        self.log_plots_tabs.addTab(self._build_log_panel(), "Log")
        self.log_plots_tabs.addTab(self._build_plots_panel(), "Plots")
        self.log_plots_tabs.addTab(self._build_peis_geis_results_panel(), "PEIS / GEIS")
        root.addWidget(self.log_plots_tabs, stretch=1)

    def _build_log_panel(self) -> QWidget:
        panel = QVBoxLayout()
        panel.addWidget(QLabel("Configuration / status"))
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        panel.addWidget(self.preview_text)
        widget = QWidget()
        widget.setLayout(panel)
        return widget

    def _build_plots_panel(self) -> QWidget:
        panel = QVBoxLayout()
        self.figure = Figure(figsize=(6, 8))
        self.canvas = FigureCanvas(self.figure)
        self._build_plot_axes(with_impedance=False)
        plots_widget, self.plots_toolbar = canvas_with_toolbar(self.canvas)
        panel.addWidget(plots_widget)
        widget = QWidget()
        widget.setLayout(panel)
        return widget

    def _build_plot_axes(self, with_impedance: bool):
        """(Re)builds the figure's subplots: potential(t) and current(t)
        always, plus a live Nyquist plot of the online per-block impedance
        when online decimation/FFT-EIS is active for the current run."""
        self.figure.clear()
        rows = 3 if with_impedance else 2
        self._ax_potential = self.figure.add_subplot(rows, 1, 1)
        self._ax_potential.set_ylabel("Potential / V")
        self._ax_potential.grid(True)
        (self._potential_line,) = self._ax_potential.plot([], [], "-", color="C0")

        self._ax_current = self.figure.add_subplot(rows, 1, 2)
        self._ax_current.set_ylabel("Current / mA")
        self._ax_current.set_xlabel("Time / s")
        self._ax_current.grid(True)
        (self._current_line,) = self._ax_current.plot([], [], "-", color="C1")

        if with_impedance:
            self._ax_impedance = self.figure.add_subplot(rows, 1, 3)
            self._ax_impedance.set_xlabel("Z' / Ohm")
            self._ax_impedance.set_ylabel("-Z'' / Ohm")
            self._ax_impedance.set_title("Latest online impedance spectrum (high-frequency, per-block FFT-EIS)")
            self._ax_impedance.grid(True)
            # Equal aspect: 1 Ohm on x == 1 Ohm on y, so a geometrically
            # circular Nyquist arc actually reads as a circle, not
            # squashed/stretched by independently-autoscaled axis ranges.
            self._ax_impedance.set_aspect("equal", adjustable="datalim")
            (self._impedance_line,) = self._ax_impedance.plot([], [], "o-", color="C2")
        else:
            self._ax_impedance = None
            self._impedance_line = None

        self.figure.tight_layout()
        self.canvas.draw_idle()
        if hasattr(self, "plots_toolbar"):
            self.plots_toolbar.update()  # a rebuilt figure invalidates the zoom history

    def _build_experiment_group(self) -> QWidget:
        group = QGroupBox("Experiment")
        form = QFormLayout(group)
        self.saving_dir_edit = QLineEdit(
            self._settings.value("paths/saving_directory", "E:/Experimental_data/")
        )
        self.saving_dir_edit.textChanged.connect(
            lambda text: self._settings.setValue("paths/saving_directory", text)
        )
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_saving_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.saving_dir_edit)
        dir_row.addWidget(browse_btn)
        dir_row_widget = QWidget()
        dir_row_widget.setLayout(dir_row)
        form.addRow("Saving directory", dir_row_widget)
        self.experiment_name_edit = QLineEdit("deis_experiment")
        form.addRow("Experiment name", self.experiment_name_edit)
        return group

    def _build_potentiostat_group(self) -> QWidget:
        group = QGroupBox("Potentiostat")
        form = QFormLayout(group)
        self.potentiostat_ip_edit = QLineEdit("USB0")
        form.addRow("Device address", self.potentiostat_ip_edit)
        # pyeclab's own bundled kernel (techniques_files_v6.04) only ships
        # VMP3-family firmware and fails to load onto newer SP-series
        # instruments ("Invalid FPGA file") -- use the real installed EC-Lab
        # Development Package instead.
        self.eclabsdk_path_edit = QLineEdit(
            self._settings.value(
                "paths/eclabsdk_path", "C:/EC-Lab Development Package/EC-Lab Development Package/lib/"
            )
        )
        self.eclabsdk_path_edit.textChanged.connect(
            lambda text: self._settings.setValue("paths/eclabsdk_path", text)
        )
        form.addRow("EC-Lab SDK path", self.eclabsdk_path_edit)
        self.potentiostat_channel_spin = QSpinBox()
        self.potentiostat_channel_spin.setRange(1, 16)
        self.potentiostat_channel_spin.setValue(1)
        form.addRow("Channel", self.potentiostat_channel_spin)
        # E range is a channel/instrument-wide voltage measurement setting,
        # not something that varies step to step -- one value for the whole
        # sequence rather than a per-technique field.
        self.potentiostat_e_range_combo = self._build_erange_combo()
        form.addRow("E range", self.potentiostat_e_range_combo)
        self.live_plot_check = QCheckBox("Live plot (embedded, in the Plots tab)")
        self.live_plot_check.setChecked(True)
        form.addRow(self.live_plot_check)
        self.external_control_check = QCheckBox("External control (AWG-driven)")
        self.external_control_check.setChecked(True)
        form.addRow(self.external_control_check)
        return group

    def _build_erange_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(_E_RANGE_OPTIONS)
        combo.setCurrentText("5 V")
        return combo

    def _build_irange_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(_I_RANGE_OPTIONS)
        combo.setCurrentText("10 mA")
        return combo

    def _build_vs_combo(self) -> QComboBox:
        """
        EC-Lab's own GUI shows a "vs." dropdown next to a step's potential/
        current value (e.g. "Ref" vs "Eoc"/"initial"), but the EC-Lab
        Development Package SDK itself only exposes ONE boolean technique
        parameter here (vs_initial) -- there's no separate, independently
        selectable "vs OCV" mode in BL_LoadTechnique's parameter list. In
        practice these end up being the same thing: a step's "initial"
        value is whatever the potentiostat reads at the moment that step
        starts, which IS the open-circuit potential whenever nothing
        earlier in the sequence has set a potential yet (e.g. a rest/OCV
        step first). So this combo offers the two real, distinct
        underlying states -- vs_initial True/False -- labeled the way
        EC-Lab's own GUI would show them, rather than inventing a third
        option the SDK has no parameter for.
        """
        combo = QComboBox()
        combo.addItems(["vs. Reference electrode", "vs. Initial value / OCV"])
        combo.setCurrentIndex(1)  # matches the old checkbox's default (checked=True)
        return combo

    @staticmethod
    def _estimate_peis_geis_sweep_time(
        initial_freq: float, final_freq: float, freq_number: int, sweep_linear: bool,
        wait_for_steady: float, average_n_times: float,
    ) -> float:
        """
        Time spent actually measuring across the sweep: (wait_for_steady +
        average_n_times) periods at every point, summed. Excludes the
        per-point overhead and the initial hold -- see
        _wire_peis_duration_estimate for those.
        """
        if freq_number <= 1 or initial_freq == final_freq:
            freqs = np.array([initial_freq])
        elif sweep_linear:
            freqs = np.linspace(initial_freq, final_freq, freq_number)
        else:
            freqs = np.logspace(np.log10(max(initial_freq, 1e-9)), np.log10(max(final_freq, 1e-9)), freq_number)
        freqs = np.maximum(freqs, 1e-9)
        return float(np.sum((wait_for_steady + average_n_times) / freqs))

    # Observed on this SP-200: successive frequency points arrived ~3.4-4.5 s
    # apart even at 100-50 kHz, where the measuring itself takes ~0.1-0.2 s --
    # i.e. a roughly constant per-point overhead dominates fast points.
    _PEIS_GEIS_PER_POINT_OVERHEAD_S = 3.5

    def _wire_peis_duration_estimate(self, initial_spin, final_spin, freq_number_spin, sweep_combo,
                                      wait_spin, avg_spin, duration_spin, label):
        def update():
            n = freq_number_spin.value()
            measuring = self._estimate_peis_geis_sweep_time(
                initial_spin.value(), final_spin.value(), n,
                sweep_combo.currentText() == "Linear", wait_spin.value(), avg_spin.value(),
            )
            hold = duration_spin.value()
            overhead = self._PEIS_GEIS_PER_POINT_OVERHEAD_S * n
            label.setText(
                f"Rough total ~{hold + measuring + overhead:.3g} s: {hold:g} s initial hold "
                f"(Duration above) + ~{measuring:.3g} s measuring + ~{overhead:.3g} s per-point "
                f"overhead ({self._PEIS_GEIS_PER_POINT_OVERHEAD_S:g} s x {n}, observed on this "
                "hardware). The first impedance point can't appear before the hold ends."
            )
        for w in (initial_spin, final_spin, freq_number_spin, wait_spin, avg_spin, duration_spin):
            w.valueChanged.connect(update)
        sweep_combo.currentTextChanged.connect(update)
        update()

    def _build_bandwidth_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(_BANDWIDTH_OPTIONS)
        # Max available (BW9) by default: higher EC-Lab bandwidth numbers
        # trade stability for a faster/wider current-follower response --
        # the ceiling is what gives the best shot at resolving the high
        # band's top frequencies (its roll-off is the leading suspect for
        # the accuracy loss seen there, see the 7-decade split test).
        combo.setCurrentText(_BANDWIDTH_OPTIONS[-1])
        return combo

    def _add_limit_fields(self, form: QFormLayout, prefix: str):
        """Adds the Ewe/I + sign + logic + value fields WithLimits techniques need."""
        type_combo = QComboBox()
        type_combo.addItems(_LIMIT_TYPE_OPTIONS)
        sign_combo = QComboBox()
        sign_combo.addItems(_LIMIT_SIGN_OPTIONS)
        logic_combo = QComboBox()
        logic_combo.addItems(_LIMIT_LOGIC_OPTIONS)
        value_spin = QDoubleSpinBox()
        value_spin.setRange(-1e6, 1e6)
        value_spin.setDecimals(6)
        setattr(self, f"{prefix}_limit_type_combo", type_combo)
        setattr(self, f"{prefix}_limit_sign_combo", sign_combo)
        setattr(self, f"{prefix}_limit_logic_combo", logic_combo)
        setattr(self, f"{prefix}_limit_value_spin", value_spin)
        form.addRow("Limit variable", type_combo)
        form.addRow("Limit sign", sign_combo)
        form.addRow("Limit logic", logic_combo)
        form.addRow("Limit value", value_spin)

    def _build_sequence_group(self) -> QWidget:
        group = QGroupBox("Technique sequence")
        layout = QVBoxLayout(group)

        self.technique_type_combo = QComboBox()
        self.technique_type_combo.addItems([
            "Chrono-Amperometry", "Chrono-Amperometry w/ Limit",
            "Chrono-Potentiometry", "Chrono-Potentiometry w/ Limit",
            "Open Circuit Voltage", "PEIS", "GEIS", "Loop",
        ])
        layout.addWidget(self.technique_type_combo)

        self.technique_stack = QStackedWidget()

        # -- Chrono-Amperometry --
        ca_widget = QWidget()
        ca_form = QFormLayout(ca_widget)
        self.ca_voltage_spin = QDoubleSpinBox()
        self.ca_voltage_spin.setRange(-10, 10)
        self.ca_voltage_spin.setDecimals(4)
        self.ca_duration_spin = QDoubleSpinBox()
        self.ca_duration_spin.setRange(0, 1e7)
        self.ca_duration_spin.setValue(30)
        self.ca_vs_init_combo = self._build_vs_combo()
        self.ca_nb_steps_spin = QSpinBox()
        self.ca_nb_steps_spin.setRange(0, 99)
        self.ca_record_dt_spin = QDoubleSpinBox()
        self.ca_record_dt_spin.setRange(0, 1e6)
        self.ca_record_dt_spin.setValue(1)
        self.ca_record_dI_spin = QDoubleSpinBox()
        self.ca_record_dI_spin.setRange(0, 1e6)
        self.ca_record_dI_spin.setValue(1)
        self.ca_repeat_spin = QSpinBox()
        self.ca_repeat_spin.setRange(0, 999)
        self.ca_i_range_combo = self._build_irange_combo()
        self.ca_bandwidth_combo = self._build_bandwidth_combo()
        ca_form.addRow("Voltage (V)", self.ca_voltage_spin)
        ca_form.addRow("Duration (s)", self.ca_duration_spin)
        ca_form.addRow("vs.", self.ca_vs_init_combo)
        ca_form.addRow("Number of steps", self.ca_nb_steps_spin)
        ca_form.addRow("Record every dt (s)", self.ca_record_dt_spin)
        ca_form.addRow("Record every dI (mA)", self.ca_record_dI_spin)
        ca_form.addRow("Repeat (cycles)", self.ca_repeat_spin)
        ca_form.addRow("I range", self.ca_i_range_combo)
        ca_form.addRow("Bandwidth", self.ca_bandwidth_combo)
        self.technique_stack.addWidget(ca_widget)

        # -- Chrono-Amperometry with Limit --
        calim_widget = QWidget()
        calim_form = QFormLayout(calim_widget)
        self.calim_voltage_spin = QDoubleSpinBox()
        self.calim_voltage_spin.setRange(-10, 10)
        self.calim_voltage_spin.setDecimals(4)
        self.calim_duration_spin = QDoubleSpinBox()
        self.calim_duration_spin.setRange(0, 1e7)
        self.calim_duration_spin.setValue(30)
        self.calim_vs_init_combo = self._build_vs_combo()
        self.calim_nb_steps_spin = QSpinBox()
        self.calim_nb_steps_spin.setRange(0, 99)
        self.calim_record_dt_spin = QDoubleSpinBox()
        self.calim_record_dt_spin.setRange(0, 1e6)
        self.calim_record_dt_spin.setValue(1)
        self.calim_record_dI_spin = QDoubleSpinBox()
        self.calim_record_dI_spin.setRange(0, 1e6)
        self.calim_record_dI_spin.setValue(1)
        self.calim_repeat_spin = QSpinBox()
        self.calim_repeat_spin.setRange(0, 999)
        self.calim_i_range_combo = self._build_irange_combo()
        self.calim_bandwidth_combo = self._build_bandwidth_combo()
        calim_form.addRow("Voltage (V)", self.calim_voltage_spin)
        calim_form.addRow("Duration (s)", self.calim_duration_spin)
        calim_form.addRow("vs.", self.calim_vs_init_combo)
        calim_form.addRow("Number of steps", self.calim_nb_steps_spin)
        calim_form.addRow("Record every dt (s)", self.calim_record_dt_spin)
        calim_form.addRow("Record every dI (mA)", self.calim_record_dI_spin)
        calim_form.addRow("Repeat (cycles)", self.calim_repeat_spin)
        calim_form.addRow("I range", self.calim_i_range_combo)
        calim_form.addRow("Bandwidth", self.calim_bandwidth_combo)
        self._add_limit_fields(calim_form, "calim")
        self.technique_stack.addWidget(calim_widget)

        # -- Chrono-Potentiometry --
        cp_widget = QWidget()
        cp_form = QFormLayout(cp_widget)
        self.cp_current_spin = QDoubleSpinBox()
        self.cp_current_spin.setRange(-10, 10)
        self.cp_current_spin.setDecimals(6)
        self.cp_duration_spin = QDoubleSpinBox()
        self.cp_duration_spin.setRange(0, 1e7)
        self.cp_duration_spin.setValue(30)
        self.cp_vs_init_combo = self._build_vs_combo()
        self.cp_nb_steps_spin = QSpinBox()
        self.cp_nb_steps_spin.setRange(0, 99)
        self.cp_record_dt_spin = QDoubleSpinBox()
        self.cp_record_dt_spin.setRange(0, 1e6)
        self.cp_record_dt_spin.setValue(1)
        self.cp_record_dE_spin = QDoubleSpinBox()
        self.cp_record_dE_spin.setRange(0, 1e6)
        self.cp_record_dE_spin.setValue(1)
        self.cp_repeat_spin = QSpinBox()
        self.cp_repeat_spin.setRange(0, 999)
        self.cp_i_range_combo = self._build_irange_combo()
        self.cp_i_range_combo.setCurrentText("10 mA")
        self.cp_bandwidth_combo = self._build_bandwidth_combo()
        cp_form.addRow("Current (A)", self.cp_current_spin)
        cp_form.addRow("Duration (s)", self.cp_duration_spin)
        cp_form.addRow("vs.", self.cp_vs_init_combo)
        cp_form.addRow("Number of steps", self.cp_nb_steps_spin)
        cp_form.addRow("Record every dt (s)", self.cp_record_dt_spin)
        cp_form.addRow("Record every dE (mV)", self.cp_record_dE_spin)
        cp_form.addRow("Repeat (cycles)", self.cp_repeat_spin)
        self.cp_i_range_combo.setToolTip("Must not be Auto for this technique.")
        cp_form.addRow("I range", self.cp_i_range_combo)
        cp_form.addRow("Bandwidth", self.cp_bandwidth_combo)
        self.technique_stack.addWidget(cp_widget)

        # -- Chrono-Potentiometry with Limit --
        cplim_widget = QWidget()
        cplim_form = QFormLayout(cplim_widget)
        self.cplim_current_spin = QDoubleSpinBox()
        self.cplim_current_spin.setRange(-10, 10)
        self.cplim_current_spin.setDecimals(6)
        self.cplim_duration_spin = QDoubleSpinBox()
        self.cplim_duration_spin.setRange(0, 1e7)
        self.cplim_duration_spin.setValue(30)
        self.cplim_vs_init_combo = self._build_vs_combo()
        self.cplim_nb_steps_spin = QSpinBox()
        self.cplim_nb_steps_spin.setRange(0, 99)
        self.cplim_record_dt_spin = QDoubleSpinBox()
        self.cplim_record_dt_spin.setRange(0, 1e6)
        self.cplim_record_dt_spin.setValue(1)
        self.cplim_record_dE_spin = QDoubleSpinBox()
        self.cplim_record_dE_spin.setRange(0, 1e6)
        self.cplim_record_dE_spin.setValue(1)
        self.cplim_repeat_spin = QSpinBox()
        self.cplim_repeat_spin.setRange(0, 999)
        self.cplim_i_range_combo = self._build_irange_combo()
        self.cplim_i_range_combo.setCurrentText("10 mA")
        self.cplim_bandwidth_combo = self._build_bandwidth_combo()
        cplim_form.addRow("Current (A)", self.cplim_current_spin)
        cplim_form.addRow("Duration (s)", self.cplim_duration_spin)
        cplim_form.addRow("vs.", self.cplim_vs_init_combo)
        cplim_form.addRow("Number of steps", self.cplim_nb_steps_spin)
        cplim_form.addRow("Record every dt (s)", self.cplim_record_dt_spin)
        cplim_form.addRow("Record every dE (mV)", self.cplim_record_dE_spin)
        cplim_form.addRow("Repeat (cycles)", self.cplim_repeat_spin)
        self.cplim_i_range_combo.setToolTip("Must not be Auto for this technique.")
        cplim_form.addRow("I range", self.cplim_i_range_combo)
        cplim_form.addRow("Bandwidth", self.cplim_bandwidth_combo)
        self._add_limit_fields(cplim_form, "cplim")
        self.technique_stack.addWidget(cplim_widget)

        # -- Open Circuit Voltage --
        ocv_widget = QWidget()
        ocv_form = QFormLayout(ocv_widget)
        self.ocv_duration_spin = QDoubleSpinBox()
        self.ocv_duration_spin.setRange(0, 1e7)
        self.ocv_duration_spin.setValue(30)
        self.ocv_record_dt_spin = QDoubleSpinBox()
        self.ocv_record_dt_spin.setRange(0, 1e6)
        self.ocv_record_dt_spin.setValue(1)
        self.ocv_bandwidth_combo = self._build_bandwidth_combo()
        ocv_form.addRow("Duration (s)", self.ocv_duration_spin)
        ocv_form.addRow("Record every dt (s)", self.ocv_record_dt_spin)
        ocv_form.addRow("Bandwidth", self.ocv_bandwidth_combo)
        self.technique_stack.addWidget(ocv_widget)

        # -- PEIS -- (same ~13 parameters as the standalone "PEIS / GEIS
        # diagnostic" group, but dedicated widgets -- this tab's own
        # convention is one widget set per sequence-step type, even where
        # parameters overlap (see ca_* vs calim_*), so a step can be
        # edited/saved/restored independently of the standalone group.
        peis_widget = QWidget()
        peis_form = QFormLayout(peis_widget)
        self.peis_vs_init_combo = self._build_vs_combo()
        peis_form.addRow("vs.", self.peis_vs_init_combo)
        self.peis_voltage_step_spin = QDoubleSpinBox()
        self.peis_voltage_step_spin.setRange(-10, 10)
        self.peis_voltage_step_spin.setDecimals(6)
        peis_form.addRow("Initial voltage step (V)", self.peis_voltage_step_spin)
        self.peis_amplitude_spin = QDoubleSpinBox()
        self.peis_amplitude_spin.setRange(0, 10)
        self.peis_amplitude_spin.setDecimals(6)
        self.peis_amplitude_spin.setValue(0.05)
        peis_form.addRow("Sine amplitude (V)", self.peis_amplitude_spin)
        self.peis_initial_freq_spin = QDoubleSpinBox()
        self.peis_initial_freq_spin.setRange(1e-6, 1e7)
        self.peis_initial_freq_spin.setDecimals(6)
        self.peis_initial_freq_spin.setValue(100_000.0)
        peis_form.addRow("Initial frequency (Hz)", self.peis_initial_freq_spin)
        self.peis_final_freq_spin = QDoubleSpinBox()
        self.peis_final_freq_spin.setRange(1e-6, 1e7)
        self.peis_final_freq_spin.setDecimals(6)
        self.peis_final_freq_spin.setValue(1.0)
        peis_form.addRow("Final frequency (Hz)", self.peis_final_freq_spin)
        self.peis_sweep_combo = QComboBox()
        self.peis_sweep_combo.addItems(["Logarithmic", "Linear"])
        peis_form.addRow("Point spacing", self.peis_sweep_combo)
        self.peis_freq_number_spin = QSpinBox()
        self.peis_freq_number_spin.setRange(1, 1000)
        self.peis_freq_number_spin.setValue(10)
        peis_form.addRow("Number of frequencies", self.peis_freq_number_spin)
        self.peis_average_n_spin = QSpinBox()
        self.peis_average_n_spin.setRange(1, 1000)
        self.peis_average_n_spin.setValue(3)
        peis_form.addRow("Average N times", self.peis_average_n_spin)
        self.peis_wait_steady_spin = QDoubleSpinBox()
        self.peis_wait_steady_spin.setRange(0, 1e6)
        self.peis_wait_steady_spin.setValue(1.0)
        peis_form.addRow("Wait for steady (periods)", self.peis_wait_steady_spin)
        self.peis_correction_check = QCheckBox("Non-stationary correction")
        self.peis_correction_check.setChecked(True)
        peis_form.addRow(self.peis_correction_check)
        self.peis_duration_step_spin = QDoubleSpinBox()
        self.peis_duration_step_spin.setRange(0, 1e7)
        self.peis_duration_step_spin.setValue(60.0)
        peis_form.addRow(with_help(
            self.peis_duration_step_spin,
            "How long the initial potential is HELD before the frequency sweep "
            "starts (EC-Lab's step duration, process 0) -- NOT a cutoff or "
            "timeout for the sweep. The first impedance point can't appear until "
            "this has elapsed, and the Potential/Current-vs-time plot only "
            "covers this hold (the sweep itself is shown in the PEIS / GEIS "
            "tab). Shorten it for a quick check; lengthen it to let a real "
            "cell settle at the step potential first. Also what this step "
            "contributes to the oscilloscope capture-size calculation when "
            "online decimation is off."
        ))
        self.peis_duration_estimate_label = QLabel()
        self.peis_duration_estimate_label.setStyleSheet("color: gray;")
        self.peis_duration_estimate_label.setWordWrap(True)
        peis_form.addRow(self.peis_duration_estimate_label)
        self.peis_record_dt_spin = QDoubleSpinBox()
        self.peis_record_dt_spin.setRange(0, 1e6)
        self.peis_record_dt_spin.setValue(0.0)
        peis_form.addRow("Record every dT (s)", self.peis_record_dt_spin)
        self.peis_record_dI_spin = QDoubleSpinBox()
        self.peis_record_dI_spin.setRange(0, 1e6)
        self.peis_record_dI_spin.setValue(0.0)
        peis_form.addRow("Record every dI (A)", self.peis_record_dI_spin)
        self.peis_external_control_check = QCheckBox("Enable external control (AWG) for this step")
        self.peis_external_control_check.setChecked(False)
        peis_form.addRow(with_help(
            self.peis_external_control_check,
            "Left OFF by default regardless of the Potentiostat section's "
            "own External control setting -- PEIS measures through the "
            "potentiostat's own internal sine generator; turning this on "
            "would route the AWG's signal into the measurement too."
        ))
        self._wire_peis_duration_estimate(
            self.peis_initial_freq_spin, self.peis_final_freq_spin, self.peis_freq_number_spin,
            self.peis_sweep_combo, self.peis_wait_steady_spin, self.peis_average_n_spin,
            self.peis_duration_step_spin, self.peis_duration_estimate_label,
        )
        self.technique_stack.addWidget(peis_widget)

        # -- GEIS -- same shape as PEIS, current/I_Range instead of voltage.
        geis_widget = QWidget()
        geis_form = QFormLayout(geis_widget)
        self.geis_vs_init_combo = self._build_vs_combo()
        geis_form.addRow("vs.", self.geis_vs_init_combo)
        self.geis_current_step_spin = QDoubleSpinBox()
        self.geis_current_step_spin.setRange(-10, 10)
        self.geis_current_step_spin.setDecimals(6)
        geis_form.addRow("Initial current step (A)", self.geis_current_step_spin)
        self.geis_amplitude_spin = QDoubleSpinBox()
        self.geis_amplitude_spin.setRange(0, 10)
        self.geis_amplitude_spin.setDecimals(6)
        geis_form.addRow("Sine amplitude (A)", self.geis_amplitude_spin)
        self.geis_initial_freq_spin = QDoubleSpinBox()
        self.geis_initial_freq_spin.setRange(1e-6, 1e7)
        self.geis_initial_freq_spin.setDecimals(6)
        self.geis_initial_freq_spin.setValue(100_000.0)
        geis_form.addRow("Initial frequency (Hz)", self.geis_initial_freq_spin)
        self.geis_final_freq_spin = QDoubleSpinBox()
        self.geis_final_freq_spin.setRange(1e-6, 1e7)
        self.geis_final_freq_spin.setDecimals(6)
        self.geis_final_freq_spin.setValue(1.0)
        geis_form.addRow("Final frequency (Hz)", self.geis_final_freq_spin)
        self.geis_sweep_combo = QComboBox()
        self.geis_sweep_combo.addItems(["Logarithmic", "Linear"])
        geis_form.addRow("Point spacing", self.geis_sweep_combo)
        self.geis_freq_number_spin = QSpinBox()
        self.geis_freq_number_spin.setRange(1, 1000)
        self.geis_freq_number_spin.setValue(10)
        geis_form.addRow("Number of frequencies", self.geis_freq_number_spin)
        self.geis_average_n_spin = QSpinBox()
        self.geis_average_n_spin.setRange(1, 1000)
        self.geis_average_n_spin.setValue(3)
        geis_form.addRow("Average N times", self.geis_average_n_spin)
        self.geis_wait_steady_spin = QDoubleSpinBox()
        self.geis_wait_steady_spin.setRange(0, 1e6)
        self.geis_wait_steady_spin.setValue(1.0)
        geis_form.addRow("Wait for steady (periods)", self.geis_wait_steady_spin)
        self.geis_correction_check = QCheckBox("Non-stationary correction")
        self.geis_correction_check.setChecked(True)
        geis_form.addRow(self.geis_correction_check)
        self.geis_duration_step_spin = QDoubleSpinBox()
        self.geis_duration_step_spin.setRange(0, 1e7)
        self.geis_duration_step_spin.setValue(60.0)
        geis_form.addRow(with_help(
            self.geis_duration_step_spin,
            "How long the initial current is HELD before the frequency sweep "
            "starts (EC-Lab's step duration, process 0) -- NOT a cutoff or "
            "timeout for the sweep. The first impedance point can't appear until "
            "this has elapsed, and the Potential/Current-vs-time plot only "
            "covers this hold (the sweep itself is shown in the PEIS / GEIS "
            "tab). Shorten it for a quick check; lengthen it to let a real "
            "cell settle at the step current first. Also what this step "
            "contributes to the oscilloscope capture-size calculation when "
            "online decimation is off."
        ))
        self.geis_duration_estimate_label = QLabel()
        self.geis_duration_estimate_label.setStyleSheet("color: gray;")
        self.geis_duration_estimate_label.setWordWrap(True)
        geis_form.addRow(self.geis_duration_estimate_label)
        self.geis_record_dt_spin = QDoubleSpinBox()
        self.geis_record_dt_spin.setRange(0, 1e6)
        self.geis_record_dt_spin.setValue(0.0)
        geis_form.addRow("Record every dT (s)", self.geis_record_dt_spin)
        self.geis_record_dE_spin = QDoubleSpinBox()
        self.geis_record_dE_spin.setRange(0, 1e6)
        self.geis_record_dE_spin.setValue(0.0)
        geis_form.addRow("Record every dE (V)", self.geis_record_dE_spin)
        self.geis_i_range_combo = self._build_irange_combo()
        geis_form.addRow("I range", self.geis_i_range_combo)
        self.geis_external_control_check = QCheckBox("Enable external control (AWG) for this step")
        self.geis_external_control_check.setChecked(False)
        geis_form.addRow(with_help(
            self.geis_external_control_check,
            "Left OFF by default regardless of the Potentiostat section's "
            "own External control setting -- GEIS measures through the "
            "potentiostat's own internal sine generator; turning this on "
            "would route the AWG's signal into the measurement too."
        ))
        self._wire_peis_duration_estimate(
            self.geis_initial_freq_spin, self.geis_final_freq_spin, self.geis_freq_number_spin,
            self.geis_sweep_combo, self.geis_wait_steady_spin, self.geis_average_n_spin,
            self.geis_duration_step_spin, self.geis_duration_estimate_label,
        )
        self.technique_stack.addWidget(geis_widget)

        # -- Loop --
        loop_widget = QWidget()
        loop_form = QFormLayout(loop_widget)
        self.loop_repeat_spin = QSpinBox()
        self.loop_repeat_spin.setRange(0, 99999)
        self.loop_repeat_spin.setValue(1)
        self.loop_start_spin = QSpinBox()
        self.loop_start_spin.setRange(0, 99)
        loop_form.addRow("Repeat N times", self.loop_repeat_spin)
        loop_form.addRow("Loop back to step #", self.loop_start_spin)
        self.technique_stack.addWidget(loop_widget)

        self.technique_type_combo.currentIndexChanged.connect(self.technique_stack.setCurrentIndex)
        layout.addWidget(self.technique_stack)

        add_remove_row = QHBoxLayout()
        add_btn = QPushButton("Add step")
        add_btn.clicked.connect(self._on_add_technique_step)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_technique_step)
        add_remove_row.addWidget(add_btn)
        add_remove_row.addWidget(remove_btn)
        add_remove_row_widget = QWidget()
        add_remove_row_widget.setLayout(add_remove_row)
        layout.addWidget(add_remove_row_widget)

        layout.addWidget(help_row(
            "Tick a step to run DEIS during it: the AWG multisine is on and the "
            "oscilloscope data are analysed and saved. Untick it and the AWG is off "
            "for that step and the oscilloscope data are discarded (the potentiostat "
            "measures regardless). New CA/CP steps start ticked; OCV, PEIS and GEIS "
            "steps start unticked. With online analysis switched off the oscilloscope "
            "captures the whole run in one shot, so there only the AWG follows the "
            "ticks."
        ))
        self.sequence_list = QListWidget()
        self.sequence_list.itemChanged.connect(self._on_sequence_item_changed)
        layout.addWidget(self.sequence_list)
        return group

    def _add_sequence_list_item(self, spec: dict):
        """Appends the list row for `spec`: a checkbox (the DEIS flag) for every step but Loop."""
        item = QListWidgetItem(f"{self.sequence_list.count() + 1}. {self._sequence_spec_label(spec)}")
        if spec["type"] == "Loop":
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
        else:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if _spec_deis_enabled(spec) else Qt.Unchecked)
            item.setToolTip("Tick: DEIS (AWG + oscilloscope) is on during this step")
        was_blocked = self.sequence_list.blockSignals(True)
        try:
            self.sequence_list.addItem(item)
        finally:
            self.sequence_list.blockSignals(was_blocked)

    def _on_sequence_item_changed(self, item):
        row = self.sequence_list.row(item)
        if not 0 <= row < len(self._sequence_specs):
            return
        spec = self._sequence_specs[row]
        if spec["type"] != "Loop":
            spec["deis"] = item.checkState() == Qt.Checked

    def _build_conditions_group(self) -> QWidget:
        group = QGroupBox("Software limits (rolling average)")
        layout = QVBoxLayout(group)
        layout.addWidget(help_row(
            "Ends a step early once the average of a quantity over the given "
            "window (checked once per second, so the window is in seconds) "
            "crosses a threshold -- a softer, averaged alternative/backup to "
            "a WithLimits step's instantaneous hardware limit."
        ))

        form = QFormLayout()
        self.condition_step_spin = QSpinBox()
        self.condition_step_spin.setRange(0, 9999)
        form.addRow("Sequence step # (0-based, as listed above)", self.condition_step_spin)
        self.condition_quantity_combo = QComboBox()
        self.condition_quantity_combo.addItems(_CONDITION_QUANTITY_OPTIONS)
        form.addRow("Quantity", self.condition_quantity_combo)
        self.condition_sign_combo = QComboBox()
        self.condition_sign_combo.addItems(_LIMIT_SIGN_OPTIONS)
        form.addRow("Sign", self.condition_sign_combo)
        self.condition_threshold_spin = QDoubleSpinBox()
        self.condition_threshold_spin.setRange(-1e6, 1e6)
        self.condition_threshold_spin.setDecimals(6)
        form.addRow("Threshold", self.condition_threshold_spin)
        self.condition_window_spin = QSpinBox()
        self.condition_window_spin.setRange(1, 100_000)
        self.condition_window_spin.setValue(60)
        form.addRow("Averaging window (s)", self.condition_window_spin)
        form_widget = QWidget()
        form_widget.setLayout(form)
        layout.addWidget(form_widget)

        add_remove_row = QHBoxLayout()
        add_btn = QPushButton("Add condition")
        add_btn.clicked.connect(self._on_add_condition)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_condition)
        add_remove_row.addWidget(add_btn)
        add_remove_row.addWidget(remove_btn)
        add_remove_row_widget = QWidget()
        add_remove_row_widget.setLayout(add_remove_row)
        layout.addWidget(add_remove_row_widget)

        self.conditions_list = QListWidget()
        layout.addWidget(self.conditions_list)
        return group

    def _build_peis_geis_results_panel(self) -> QWidget:
        """
        Impedance results of any PEIS/GEIS steps in the technique sequence:
        points are drawn live as each frequency's row arrives (see
        _update_peis_geis_live_plot), and written to the experiment folder
        when the run ends or is stopped (see _save_peis_geis_run_results).
        """
        panel = QVBoxLayout()
        self.peis_geis_status_label = QLabel("No PEIS/GEIS step run yet -- add one to the technique sequence.")
        self.peis_geis_status_label.setStyleSheet("color: gray;")
        self.peis_geis_status_label.setWordWrap(True)
        panel.addWidget(self.peis_geis_status_label)

        self.peis_geis_figure = Figure(figsize=(8, 7))
        self.peis_geis_canvas = FigureCanvas(self.peis_geis_figure)
        self._build_peis_geis_axes()
        peis_geis_plot_widget, self.peis_geis_toolbar = canvas_with_toolbar(self.peis_geis_canvas)
        panel.addWidget(peis_geis_plot_widget)

        save_row = QHBoxLayout()
        self.peis_geis_save_btn = QPushButton("Save a copy...")
        self.peis_geis_save_btn.clicked.connect(self._on_save_peis_geis_results)
        self.peis_geis_save_btn.setEnabled(False)
        save_row.addWidget(self.peis_geis_save_btn)
        save_row_widget = QWidget()
        save_row_widget.setLayout(save_row)
        panel.addWidget(save_row_widget)

        self.peis_geis_results_text = QPlainTextEdit()
        self.peis_geis_results_text.setReadOnly(True)
        self.peis_geis_results_text.setMaximumHeight(160)
        panel.addWidget(self.peis_geis_results_text)

        widget = QWidget()
        widget.setLayout(panel)
        self._peis_geis_result = None  # (freqs, z, step_index) of the last run that produced points
        self._peis_live_rows_shown = 0  # reset per run, see _on_start_experiment
        return widget

    def _build_peis_geis_axes(self):
        self.peis_geis_figure.clear()
        self._ax_peis_nyquist = self.peis_geis_figure.add_subplot(1, 2, 1)
        self._ax_peis_nyquist.set_xlabel("Z' / Ohm")
        self._ax_peis_nyquist.set_ylabel("-Z'' / Ohm")
        self._ax_peis_nyquist.set_title("Nyquist")
        self._ax_peis_nyquist.grid(True)
        self._ax_peis_bode = self.peis_geis_figure.add_subplot(1, 2, 2)
        self._ax_peis_bode.set_xlabel("Frequency / Hz")
        self._ax_peis_bode.set_ylabel("|Z| / Ohm")
        self._ax_peis_bode.set_xscale("log")
        self._ax_peis_bode.set_yscale("log")
        self._ax_peis_bode.set_title("Bode (magnitude)")
        self._ax_peis_bode.grid(True, which="both")
        self.peis_geis_figure.tight_layout()

    def _on_save_peis_geis_results(self):
        """Save-a-copy of the last run's results to a location of choice --
        the run itself already wrote peis_geis_results.csv into its own
        experiment folder (see _save_peis_geis_run_results)."""
        if self._peis_geis_result is None:
            return
        freqs, z, step_index = self._peis_geis_result
        start_dir = self._settings.value("paths/last_settings_dir", "")
        path, _ = QFileDialog.getSaveFileName(self, "Save PEIS/GEIS results", start_dir, "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            self._write_peis_geis_csv(path, freqs, z, step_index)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save results", f"{type(exc).__name__}: {exc}")
            return
        self.peis_geis_status_label.setText(f"Saved a copy of {freqs.size} point(s) to {path}")

    @staticmethod
    def _write_peis_geis_csv(path, freqs, z, step_index):
        write_peis_geis_csv(path, freqs, z, step_index)

    def _save_peis_geis_run_results(self):
        """
        Called when a run finishes or is stopped: collects whatever PEIS/GEIS
        points the channel received, writes peis_geis_results.csv into
        the experiment folder, and enables the "Save a copy" button. A
        no-op for runs without a PEIS/GEIS step. The raw rows live only in
        PEISAwareChannel.peis_* (they are deliberately not part of
        measurement_data.txt, which is a (t, Ewe, I) stream), so without
        this they'd vanish with the run.
        """
        pts = self._peis_geis_channel_points()
        if pts is None:
            return
        freqs, z, step_index = pts
        self._peis_geis_result = pts
        folder = Path(self.saving_dir_edit.text()) / self.experiment_name_edit.text()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self._write_peis_geis_csv(folder / "peis_geis_results.csv", freqs, z, step_index)
            where = f"Saved to {folder}"
        except Exception as exc:
            where = f"COULD NOT SAVE to {folder}: {type(exc).__name__}: {exc}"
        self.peis_geis_status_label.setText(f"{freqs.size} frequency point(s) from this run. {where}")
        self.peis_geis_save_btn.setEnabled(True)

    def _build_awg_group(self) -> QWidget:
        group = QGroupBox("Waveform generator (AWG)")
        form = QFormLayout(group)
        self.awg_enabled_check = QCheckBox("Enable AWG-driven perturbation")
        self.awg_enabled_check.setChecked(True)
        form.addRow(self.awg_enabled_check)
        self.awg_address_edit = QLineEdit("USB0::0x0957::0x2C07::MY62003024::0::INSTR")
        form.addRow("VISA address", self.awg_address_edit)
        self.awg_channel_spin = QSpinBox()
        self.awg_channel_spin.setRange(1, 2)
        form.addRow("AWG channel", self.awg_channel_spin)
        load_btn = QPushButton("Load multisine design (JSON)...")
        load_btn.clicked.connect(self._on_load_multisine)
        form.addRow(load_btn)
        self.multisine_status_label = QLabel("No multisine loaded.")
        self.multisine_status_label.setStyleSheet("color: gray;")
        self.multisine_status_label.setWordWrap(True)
        form.addRow(self.multisine_status_label)
        base_freq_row = QHBoxLayout()
        self.base_frequency_spin = QDoubleSpinBox()
        self.base_frequency_spin.setRange(1e-9, 1e9)
        self.base_frequency_spin.setDecimals(6)
        self.base_frequency_spin.setValue(1.0)
        self.base_frequency_unit_combo = QComboBox()
        self.base_frequency_unit_combo.addItems(_FREQ_UNIT_OPTIONS)
        self.base_frequency_unit_combo.setCurrentText("Hz")
        base_freq_row.addWidget(self.base_frequency_spin)
        base_freq_row.addWidget(self.base_frequency_unit_combo)
        base_freq_row.addWidget(HelpButton(
            "The Multisine Designer works in harmonic units (1st harmonic = 1) -- this "
            "rescales the loaded design's frequencies and the AWG sample rate so the 1st "
            "harmonic lands here in real Hz, every other harmonic scaled the same way. "
            "The underlying waveform samples are unchanged, only their timing -- takes "
            "effect immediately, no need to reload the file."
        ))
        self.base_frequency_spin.valueChanged.connect(self._apply_base_frequency)
        self.base_frequency_unit_combo.currentTextChanged.connect(self._apply_base_frequency)
        form.addRow("Base frequency (1st harmonic)", base_freq_row)
        self.awg_amplitude_type_combo = QComboBox()
        self.awg_amplitude_type_combo.addItems(["Potential (V)", "Current (A)"])
        self.awg_amplitude_type_combo.currentTextChanged.connect(self._on_awg_amplitude_type_changed)
        form.addRow("Amplitude type", with_help(
            self.awg_amplitude_type_combo,
            "Current mode: converted to the voltage injected via external control, using "
            "the first sequence step's I range (must be Chrono-Potentiometry/w-Limit -- "
            "the injected voltage only maps to current in galvanostatic mode). Per the "
            "EC-Lab Development Package manual, 1 V = full scale of that I range, and the "
            "injected voltage must not exceed +/-1V -- checked at Start, and shown "
            "resolved to volts in Preview configuration."
        ))
        self.awg_amplitude_spin = QDoubleSpinBox()
        self.awg_amplitude_spin.setRange(0, 10)
        self.awg_amplitude_spin.setDecimals(6)
        self.awg_amplitude_spin.setValue(0.05)
        self.awg_amplitude_label = QLabel("Amplitude, peak-to-peak (V)")
        form.addRow(self.awg_amplitude_label, self.awg_amplitude_spin)
        return group

    def _on_awg_amplitude_type_changed(self, amplitude_type: str):
        if amplitude_type == "Current (A)":
            self.awg_amplitude_label.setText("Amplitude, peak-to-peak (A)")
        else:
            self.awg_amplitude_label.setText("Amplitude, peak-to-peak (V)")

    def _build_scope_group(self) -> QWidget:
        group = QGroupBox("Oscilloscope")
        outer = QVBoxLayout(group)

        scope_enabled_row = QHBoxLayout()
        scope_enabled_row.setContentsMargins(0, 0, 0, 0)
        self.scope_enabled_check = QCheckBox("Enable oscilloscope capture")
        self.scope_enabled_check.setChecked(True)
        self.scope_enabled_check.toggled.connect(self._on_scope_enabled_toggled)
        scope_enabled_row.addWidget(self.scope_enabled_check)
        scope_enabled_row.addWidget(HelpButton(
            "Off for a plain potentiostat-only run (e.g. OCV/PEIS/GEIS alone) with no "
            "PicoScope involved at all -- no capture size or I range needed, since those "
            "only exist to configure the scope. EC-Lab's own measurement_data.txt (at "
            "whatever Record_every_dt/dI/dE each step sets) is all that's saved. Online "
            "decimation/FFT-EIS is forced off too, since it reads from the scope."
        ))
        scope_enabled_row.addStretch(1)
        outer.addLayout(scope_enabled_row)

        self.scope_fields_widget = QWidget()
        form = QFormLayout(self.scope_fields_widget)
        form.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scope_fields_widget)

        self.scope_model_combo = QComboBox()
        self.scope_model_combo.addItems(["Picoscope5000a", "Picoscope4000"])
        form.addRow("Model", self.scope_model_combo)

        self.scope_resolution_combo = QComboBox()
        self.scope_resolution_combo.addItems(_RESOLUTION_OPTIONS)
        self.scope_resolution_combo.setCurrentText("14 bit")
        form.addRow("Resolution", self.scope_resolution_combo)

        # PicoSDK's sampling_time is passed straight into ctypes.c_int32 --
        # must be a real int, not a float (QDoubleSpinBox raised TypeError).
        self.scope_sampling_value_spin = QSpinBox()
        self.scope_sampling_value_spin.setRange(1, 1_000_000_000)
        self.scope_sampling_value_spin.setValue(1)
        self.scope_sampling_unit_combo = QComboBox()
        self.scope_sampling_unit_combo.addItems(_TIME_UNIT_OPTIONS)
        self.scope_sampling_unit_combo.setCurrentText("ms")
        sampling_row = QHBoxLayout()
        sampling_row.setContentsMargins(0, 0, 0, 0)
        sampling_row.addWidget(self.scope_sampling_value_spin)
        sampling_row.addWidget(self.scope_sampling_unit_combo)
        sampling_row_widget = QWidget()
        sampling_row_widget.setLayout(sampling_row)
        form.addRow("Sampling interval", sampling_row_widget)
        self.scope_sampling_value_spin.valueChanged.connect(self._update_analysis_auto_label)
        self.scope_sampling_unit_combo.currentTextChanged.connect(self._update_analysis_auto_label)

        self.scope_range_a_combo = QComboBox()
        self.scope_range_a_combo.addItems(_VRANGE_OPTIONS)
        self.scope_range_a_combo.setCurrentText("±1 V")
        form.addRow("Channel A range (voltage)", self.scope_range_a_combo)
        self.scope_range_b_combo = QComboBox()
        self.scope_range_b_combo.addItems(_VRANGE_OPTIONS)
        self.scope_range_b_combo.setCurrentText("±500 mV")
        form.addRow("Channel B range (current probe)", self.scope_range_b_combo)

        self.scope_conv_factor_vref_spin = QDoubleSpinBox()
        self.scope_conv_factor_vref_spin.setRange(1e-9, 1e9)
        self.scope_conv_factor_vref_spin.setDecimals(6)
        # The current-monitor output is treated as equivalent to a
        # resistor R = Vref / I_full_scale (the resistor that would turn
        # full-scale I-range current into Vref on the monitor output);
        # conv_factor = 1/R = I_full_scale / Vref. Vref=1V ("full-scale
        # I-range current reads as 1V on the monitor output") confirmed
        # against a known dummy cell (R1 1k + [R2 1k || C2 10n] + [R3
        # 3.57k || C3 2.2u]) at I_RANGE=10mA: measured impedance landed
        # within ~2-6% of the circuit's exact theoretical values at 2/10/
        # 20/50 Hz once pico.set_channel() actually received conv_factor
        # (it was previously passed as the 3rd positional arg, which is
        # signal_name, not conv_factor -- silently leaving the real
        # current conversion never applied at all, and making every
        # earlier "calibration" attempt measure raw scope voltage instead
        # of current). Re-run the dummy-cell check if this default doesn't
        # hold on your hardware.
        self.scope_conv_factor_vref_spin.setValue(1.0)
        form.addRow("Current monitor: V at full-scale I range (V)", with_help(
            self.scope_conv_factor_vref_spin,
            "The oscilloscope's actual current conversion factor is computed "
            "automatically from the first sequence step's I range: "
            "conv_factor = I_range_full_scale / (V above) -- equivalent to a "
            "resistor R = (V above) / I_range_full_scale converting full-scale "
            "current to that voltage."
        ))

        form.addRow("Capture size (auto)", help_row(
            "Capture size is computed automatically: the whole run when online "
            "decimation is off below, or 3x the analysis window (see the Online "
            "decimation / FFT-EIS section) when it's on."
        ))
        return group

    def _on_scope_enabled_toggled(self, checked: bool):
        self.scope_fields_widget.setEnabled(checked)
        if not checked:
            # Decimation/FFT-EIS reads from the scope -- can't stay on
            # without it. Re-enabling the scope doesn't restore it; the
            # user re-checks it explicitly, same as any other setting that
            # gets turned off alongside something it depends on.
            self.decimation_enabled_check.setChecked(False)
        self.decimation_enabled_check.setEnabled(checked)

    def _build_analysis_group(self) -> QWidget:
        group = QGroupBox("Online decimation / FFT-EIS")
        outer = QVBoxLayout(group)

        decimation_row = QHBoxLayout()
        decimation_row.setContentsMargins(0, 0, 0, 0)
        self.decimation_enabled_check = QCheckBox("Enable online decimation + FFT-EIS")
        self.decimation_enabled_check.setChecked(True)
        self.decimation_enabled_check.toggled.connect(self._on_decimation_enabled_toggled)
        decimation_row.addWidget(self.decimation_enabled_check)
        decimation_row.addWidget(HelpButton(
            "Sampling time for the online calculation is taken directly from the "
            "oscilloscope's sampling interval (Oscilloscope section) -- no separate "
            "field. When disabled: the scope just captures the whole run and saves "
            "raw voltage/current, no online processing."
        ))
        decimation_row.addStretch(1)
        outer.addLayout(decimation_row)

        self.analysis_fields_widget = QWidget()
        form = QFormLayout(self.analysis_fields_widget)
        form.setContentsMargins(0, 0, 0, 0)
        self.analysis_auto_label = QLabel(
            "Analysis window / buffer duration: load a multisine design to compute "
            "(auto: window = 1x the period of the lowest active frequency)."
        )
        self.analysis_auto_label.setStyleSheet("color: gray;")
        self.analysis_auto_label.setWordWrap(True)
        form.addRow(self.analysis_auto_label)
        self.filter_cutoff_spin = QDoubleSpinBox()
        self.filter_cutoff_spin.setRange(0, 1e6)
        self.filter_cutoff_spin.setValue(8)
        form.addRow("Decimation filter cutoff (Hz)", self.filter_cutoff_spin)
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 1000)
        self.filter_order_spin.setValue(8)
        form.addRow("Decimation filter order", self.filter_order_spin)
        self.resampling_freq_edit = QLineEdit("500")
        form.addRow("Resampling frequency (Hz)", self.resampling_freq_edit)
        outer.addWidget(self.analysis_fields_widget)

        return group

    def _on_decimation_enabled_toggled(self, checked: bool):
        self.analysis_fields_widget.setEnabled(checked)

    def _build_run_group(self) -> QWidget:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        save_load_row = QHBoxLayout()
        save_settings_btn = QPushButton("Save settings...")
        save_settings_btn.clicked.connect(self._on_save_settings)
        load_settings_btn = QPushButton("Load settings...")
        load_settings_btn.clicked.connect(self._on_load_settings)
        save_load_row.addWidget(save_settings_btn)
        save_load_row.addWidget(load_settings_btn)
        save_load_row_widget = QWidget()
        save_load_row_widget.setLayout(save_load_row)
        layout.addWidget(save_load_row_widget)
        preview_btn = QPushButton("Preview configuration")
        preview_btn.clicked.connect(self._on_preview_configuration)
        layout.addWidget(preview_btn)
        self.start_button = QPushButton("Start experiment")
        self.start_button.clicked.connect(self._on_start_experiment)
        layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_experiment)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        return group

    # ------------------------------------------------------------------ #
    # Actions -- experiment / sequence
    # ------------------------------------------------------------------ #

    def _on_browse_saving_dir(self):
        # Starts from the field's current value, which is itself the
        # persisted last-used saving directory -- separate from the
        # multisine-load dialog's own remembered folder (paths/last_multisine_dir).
        path = QFileDialog.getExistingDirectory(self, "Select saving directory", self.saving_dir_edit.text())
        if path:
            self.saving_dir_edit.setText(path)

    @staticmethod
    def _sequence_spec_label(spec: dict) -> str:
        t = spec["type"]
        if t == "CA":
            return f"CA  V={spec['voltage']}V  t={spec['duration']}s  repeat={spec['repeat']}"
        if t == "CALim":
            return (
                f"CALim  V={spec['voltage']}V  t={spec['duration']}s  "
                f"limit={spec['limit_type']}{spec['limit_sign']}{spec['limit_value']}"
            )
        if t == "CP":
            return f"CP  I={spec['current']}A  t={spec['duration']}s  repeat={spec['repeat']}"
        if t == "CPLim":
            return (
                f"CPLim  I={spec['current']}A  t={spec['duration']}s  "
                f"limit={spec['limit_type']}{spec['limit_sign']}{spec['limit_value']}"
            )
        if t == "OCV":
            return f"OCV  t={spec['duration']}s"
        if t == "PEIS":
            return (
                f"PEIS  {spec['initial_frequency']:g}->{spec['final_frequency']:g}Hz  "
                f"n={spec['frequency_number']}  amp={spec['amplitude']}V"
            )
        if t == "GEIS":
            return (
                f"GEIS  {spec['initial_frequency']:g}->{spec['final_frequency']:g}Hz  "
                f"n={spec['frequency_number']}  amp={spec['amplitude']}A"
            )
        return f"Loop  x{spec['repeat_N']}  from step #{spec['loop_start']}"

    def _on_add_technique_step(self):
        index = self.technique_type_combo.currentIndex()
        if index == 0:
            spec = {
                "type": "CA",
                "voltage": self.ca_voltage_spin.value(),
                "duration": self.ca_duration_spin.value(),
                "vs_init": self.ca_vs_init_combo.currentIndex() == 1,
                "nb_steps": self.ca_nb_steps_spin.value(),
                "record_dt": self.ca_record_dt_spin.value(),
                "record_dI": self.ca_record_dI_spin.value(),
                "repeat": self.ca_repeat_spin.value(),
                "i_range": self.ca_i_range_combo.currentText(),
                "bandwidth": self.ca_bandwidth_combo.currentText(),
            }
        elif index == 1:
            spec = {
                "type": "CALim",
                "voltage": self.calim_voltage_spin.value(),
                "duration": self.calim_duration_spin.value(),
                "vs_init": self.calim_vs_init_combo.currentIndex() == 1,
                "nb_steps": self.calim_nb_steps_spin.value(),
                "record_dt": self.calim_record_dt_spin.value(),
                "record_dI": self.calim_record_dI_spin.value(),
                "repeat": self.calim_repeat_spin.value(),
                "i_range": self.calim_i_range_combo.currentText(),
                "bandwidth": self.calim_bandwidth_combo.currentText(),
                "limit_type": self.calim_limit_type_combo.currentText(),
                "limit_sign": self.calim_limit_sign_combo.currentText(),
                "limit_logic": self.calim_limit_logic_combo.currentText(),
                "limit_value": self.calim_limit_value_spin.value(),
            }
        elif index == 2:
            spec = {
                "type": "CP",
                "current": self.cp_current_spin.value(),
                "duration": self.cp_duration_spin.value(),
                "vs_init": self.cp_vs_init_combo.currentIndex() == 1,
                "nb_steps": self.cp_nb_steps_spin.value(),
                "record_dt": self.cp_record_dt_spin.value(),
                "record_dE": self.cp_record_dE_spin.value(),
                "repeat": self.cp_repeat_spin.value(),
                "i_range": self.cp_i_range_combo.currentText(),
                "bandwidth": self.cp_bandwidth_combo.currentText(),
            }
        elif index == 3:
            spec = {
                "type": "CPLim",
                "current": self.cplim_current_spin.value(),
                "duration": self.cplim_duration_spin.value(),
                "vs_init": self.cplim_vs_init_combo.currentIndex() == 1,
                "nb_steps": self.cplim_nb_steps_spin.value(),
                "record_dt": self.cplim_record_dt_spin.value(),
                "record_dE": self.cplim_record_dE_spin.value(),
                "repeat": self.cplim_repeat_spin.value(),
                "i_range": self.cplim_i_range_combo.currentText(),
                "bandwidth": self.cplim_bandwidth_combo.currentText(),
                "limit_type": self.cplim_limit_type_combo.currentText(),
                "limit_sign": self.cplim_limit_sign_combo.currentText(),
                "limit_logic": self.cplim_limit_logic_combo.currentText(),
                "limit_value": self.cplim_limit_value_spin.value(),
            }
        elif index == 4:
            spec = {
                "type": "OCV",
                "duration": self.ocv_duration_spin.value(),
                "record_dt": self.ocv_record_dt_spin.value(),
                "bandwidth": self.ocv_bandwidth_combo.currentText(),
            }
        elif index == 5:
            spec = {
                "type": "PEIS",
                "vs_init": self.peis_vs_init_combo.currentIndex() == 1,
                "voltage_step": self.peis_voltage_step_spin.value(),
                "amplitude": self.peis_amplitude_spin.value(),
                "initial_frequency": self.peis_initial_freq_spin.value(),
                "final_frequency": self.peis_final_freq_spin.value(),
                "sweep_linear": self.peis_sweep_combo.currentText() == "Linear",
                "frequency_number": self.peis_freq_number_spin.value(),
                "average_n_times": self.peis_average_n_spin.value(),
                "wait_for_steady": self.peis_wait_steady_spin.value(),
                "correction": self.peis_correction_check.isChecked(),
                "duration": self.peis_duration_step_spin.value(),
                "record_dt": self.peis_record_dt_spin.value(),
                "record_dI": self.peis_record_dI_spin.value(),
                "external_control": self.peis_external_control_check.isChecked(),
            }
        elif index == 6:
            spec = {
                "type": "GEIS",
                "vs_init": self.geis_vs_init_combo.currentIndex() == 1,
                "current_step": self.geis_current_step_spin.value(),
                "amplitude": self.geis_amplitude_spin.value(),
                "initial_frequency": self.geis_initial_freq_spin.value(),
                "final_frequency": self.geis_final_freq_spin.value(),
                "sweep_linear": self.geis_sweep_combo.currentText() == "Linear",
                "frequency_number": self.geis_freq_number_spin.value(),
                "average_n_times": self.geis_average_n_spin.value(),
                "wait_for_steady": self.geis_wait_steady_spin.value(),
                "correction": self.geis_correction_check.isChecked(),
                "duration": self.geis_duration_step_spin.value(),
                "record_dt": self.geis_record_dt_spin.value(),
                "record_dE": self.geis_record_dE_spin.value(),
                "i_range": self.geis_i_range_combo.currentText(),
                "external_control": self.geis_external_control_check.isChecked(),
            }
        else:
            spec = {
                "type": "Loop",
                "repeat_N": self.loop_repeat_spin.value(),
                "loop_start": self.loop_start_spin.value(),
            }
        if spec["type"] != "Loop":
            spec["deis"] = _spec_deis_enabled(spec)
        self._sequence_specs.append(spec)
        self._add_sequence_list_item(spec)

    def _on_remove_technique_step(self):
        row = self.sequence_list.currentRow()
        if row < 0:
            return
        self.sequence_list.takeItem(row)
        del self._sequence_specs[row]
        # renumber remaining rows
        for i in range(self.sequence_list.count()):
            text = self.sequence_list.item(i).text().split(". ", 1)[-1]
            self.sequence_list.item(i).setText(f"{i + 1}. {text}")

    @staticmethod
    def _condition_spec_label(spec: dict) -> str:
        return (
            f"step {spec['technique_index']}: {spec['quantity']} {spec['operator']} "
            f"{spec['threshold']} (avg over {spec['num_elements']}s)"
        )

    def _on_add_condition(self):
        spec = {
            "technique_index": self.condition_step_spin.value(),
            "quantity": self.condition_quantity_combo.currentText(),
            "operator": self.condition_sign_combo.currentText(),
            "threshold": self.condition_threshold_spin.value(),
            "num_elements": self.condition_window_spin.value(),
        }
        self._condition_specs.append(spec)
        self.conditions_list.addItem(self._condition_spec_label(spec))

    def _on_remove_condition(self):
        row = self.conditions_list.currentRow()
        if row < 0:
            return
        self.conditions_list.takeItem(row)
        del self._condition_specs[row]

    # ------------------------------------------------------------------ #
    # Actions -- AWG
    # ------------------------------------------------------------------ #

    def _load_multisine_from_path(self, path: str):
        """
        Loads a multisine (single or split) from disk and rescales it to
        the currently configured base frequency. Raises on failure; caller
        decides how to report it. The raw parsed JSON is kept
        (self._loaded_multisine_data) so changing the base-frequency
        control later can re-rescale without re-reading the file.
        """
        data = load_multisine_json(path)
        self._loaded_multisine_data = data
        self._loaded_multisine_path = path
        self._apply_base_frequency()

    def _resolve_base_frequency_hz(self) -> float:
        return self.base_frequency_spin.value() * _FREQ_UNIT_TO_HZ[self.base_frequency_unit_combo.currentText()]

    def _apply_base_frequency(self):
        """
        Rebuilds self._loaded_multisine / _loaded_multisine_split from the
        raw loaded JSON (self._loaded_multisine_data) at the currently
        configured base frequency -- called on load and whenever the
        base-frequency control changes, so no reload is needed to see the
        effect. A design's own lowest/first-harmonic frequency (the low
        band's, for a split) is what gets remapped to the base-frequency
        field's value; every other frequency scales the same way, and for
        a split design both bands are scaled by the identical factor so
        they stay correctly related to one another.
        """
        if self._loaded_multisine_data is None:
            return
        try:
            base_hz = self._resolve_base_frequency_hz()
        except Exception:
            return
        data = self._loaded_multisine_data
        if data.get("type", "single") == "split":
            # load_split_multisine_bands, not multisine_from_dict directly:
            # the high band's number_points is NOT the default fs/f0 (see
            # design/io.py) -- this recovers the right value even for
            # older files that don't save it.
            low, high = load_split_multisine_bands(data)
            reference = low.frequencies.min()
            self._loaded_multisine = None
            self._loaded_multisine_split = {
                "low": rescale_multisine_frequency(low, base_hz, reference),
                "high": rescale_multisine_frequency(high, base_hz, reference),
            }
        else:
            ms = multisine_from_dict(data)
            reference = ms.frequencies.min()
            self._loaded_multisine_split = None
            self._loaded_multisine = rescale_multisine_frequency(ms, base_hz, reference)
        self._update_multisine_status_label()
        self._update_analysis_auto_label()

    def _update_multisine_status_label(self):
        if self._loaded_multisine_split is not None:
            low = self._loaded_multisine_split["low"]
            high = self._loaded_multisine_split["high"]
            self.multisine_status_label.setText(
                f"Loaded split design -- low band: {low.frequencies.size} harmonics, "
                f"[{low.frequencies.min():.4g}, {low.frequencies.max():.4g}] Hz, fs={low.sampling_frequency:.4g} "
                f"-> AWG channel 1; high band: {high.frequencies.size} harmonics, "
                f"[{high.frequencies.min():.4g}, {high.frequencies.max():.4g}] Hz, fs={high.sampling_frequency:.4g} "
                "-> AWG channel 2 (channels combined into channel 1's output; 'AWG channel' below is unused)."
            )
        elif self._loaded_multisine is not None:
            ms = self._loaded_multisine
            self.multisine_status_label.setText(
                f"Loaded: {ms.frequencies.size} harmonics, fs={ms.sampling_frequency:.4g}, "
                f"range [{ms.frequencies.min():.4g}, {ms.frequencies.max():.4g}]"
            )
        else:
            self.multisine_status_label.setText("No multisine loaded.")

    def _multisine_loaded(self) -> bool:
        return self._loaded_multisine is not None or self._loaded_multisine_split is not None

    def _active_frequencies(self):
        """
        Combined excitation frequency set for the currently loaded design
        (single or split), used for online-analysis (FFT-EIS frequency
        list) and capture-size sizing. None if nothing is loaded.
        """
        if self._loaded_multisine_split is not None:
            low = self._loaded_multisine_split["low"]
            high = self._loaded_multisine_split["high"]
            return np.sort(np.concatenate([low.frequencies, high.frequencies]))
        if self._loaded_multisine is not None:
            return self._loaded_multisine.frequencies
        return None

    def _multisine_n_harmonics(self):
        freqs = self._active_frequencies()
        return int(freqs.size) if freqs is not None else None

    def _on_load_multisine(self):
        start_dir = self._settings.value("paths/last_multisine_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Load multisine design", start_dir, "JSON files (*.json)")
        if not path:
            return
        self._settings.setValue("paths/last_multisine_dir", str(Path(path).parent))
        try:
            self._load_multisine_from_path(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load multisine design", str(exc))

    # ------------------------------------------------------------------ #
    # Actions -- save/load experiment settings
    # ------------------------------------------------------------------ #

    def _gather_settings_for_save(self) -> dict:
        """
        Every raw, user-editable widget value that defines an experiment --
        as opposed to _gather_configuration(), which also mixes in
        computed/resolved values (capture_size, the resolved AWG voltage,
        the resolved current conversion factor, ...) meant for hardware
        construction, not for faithfully restoring the input widgets.
        """
        return {
            "type": "elma_experiment_settings",
            "version": 1,
            "experiment": {
                "saving_directory": self.saving_dir_edit.text(),
                "experiment_name": self.experiment_name_edit.text(),
            },
            "potentiostat": {
                "address": self.potentiostat_ip_edit.text(),
                "eclabsdk_path": self.eclabsdk_path_edit.text(),
                "channel": self.potentiostat_channel_spin.value(),
                "e_range": self.potentiostat_e_range_combo.currentText(),
                "live_plot": self.live_plot_check.isChecked(),
                "external_control": self.external_control_check.isChecked(),
            },
            "sequence": list(self._sequence_specs),
            "software_limits": list(self._condition_specs),
            "awg": {
                "enabled": self.awg_enabled_check.isChecked(),
                "address": self.awg_address_edit.text(),
                "channel": self.awg_channel_spin.value(),
                "amplitude_type": self.awg_amplitude_type_combo.currentText(),
                "amplitude_input": self.awg_amplitude_spin.value(),
                "base_frequency_value": self.base_frequency_spin.value(),
                "base_frequency_unit": self.base_frequency_unit_combo.currentText(),
                "multisine_path": self._loaded_multisine_path,
            },
            "oscilloscope": {
                "enabled": self.scope_enabled_check.isChecked(),
                "model": self.scope_model_combo.currentText(),
                "resolution": self.scope_resolution_combo.currentText(),
                "sampling_value": self.scope_sampling_value_spin.value(),
                "sampling_unit": self.scope_sampling_unit_combo.currentText(),
                "range_a": self.scope_range_a_combo.currentText(),
                "range_b": self.scope_range_b_combo.currentText(),
                "conv_factor_vref": self.scope_conv_factor_vref_spin.value(),
            },
            "online_analysis": {
                "decimation_enabled": self.decimation_enabled_check.isChecked(),
                # window_size/buffer_duration are no longer raw widget values --
                # both are auto-computed from the loaded multisine's lowest
                # frequency and the scope sampling rate (see
                # _compute_analysis_window_and_buffer), so there's nothing to
                # save/restore for them here; they're already implied by the
                # multisine_path + base_frequency + scope sampling fields
                # saved elsewhere in this dict.
                "filter_cutoff": self.filter_cutoff_spin.value(),
                "filter_order": self.filter_order_spin.value(),
                "resampling_frequency": self.resampling_freq_edit.text(),
            },
        }

    def _apply_loaded_settings(self, data: dict):
        exp = data.get("experiment", {})
        if "saving_directory" in exp:
            self.saving_dir_edit.setText(exp["saving_directory"])
        if "experiment_name" in exp:
            self.experiment_name_edit.setText(exp["experiment_name"])

        pot = data.get("potentiostat", {})
        if "address" in pot:
            self.potentiostat_ip_edit.setText(pot["address"])
        if "eclabsdk_path" in pot:
            self.eclabsdk_path_edit.setText(pot["eclabsdk_path"])
        if "channel" in pot:
            self.potentiostat_channel_spin.setValue(pot["channel"])
        if "e_range" in pot:
            self.potentiostat_e_range_combo.setCurrentText(pot["e_range"])
        if "live_plot" in pot:
            self.live_plot_check.setChecked(pot["live_plot"])
        if "external_control" in pot:
            self.external_control_check.setChecked(pot["external_control"])

        self._sequence_specs = list(data.get("sequence", []))
        self.sequence_list.clear()
        for spec in self._sequence_specs:
            self._add_sequence_list_item(spec)

        self._condition_specs = list(data.get("software_limits", []))
        self.conditions_list.clear()
        for spec in self._condition_specs:
            self.conditions_list.addItem(self._condition_spec_label(spec))

        awg = data.get("awg", {})
        if "enabled" in awg:
            self.awg_enabled_check.setChecked(awg["enabled"])
        if "address" in awg:
            self.awg_address_edit.setText(awg["address"])
        if "channel" in awg:
            self.awg_channel_spin.setValue(awg["channel"])
        if "amplitude_type" in awg:
            self.awg_amplitude_type_combo.setCurrentText(awg["amplitude_type"])
        if "amplitude_input" in awg:
            self.awg_amplitude_spin.setValue(awg["amplitude_input"])
        if "base_frequency_value" in awg:
            self.base_frequency_spin.setValue(awg["base_frequency_value"])
        if "base_frequency_unit" in awg:
            self.base_frequency_unit_combo.setCurrentText(awg["base_frequency_unit"])
        multisine_path = awg.get("multisine_path")
        if multisine_path:
            try:
                self._load_multisine_from_path(multisine_path)
            except Exception as exc:
                self._loaded_multisine = None
                self._loaded_multisine_split = None
                self._loaded_multisine_data = None
                self._loaded_multisine_path = None
                self.multisine_status_label.setText(
                    f"Could not reload multisine from '{multisine_path}': {exc}"
                )

        scope = data.get("oscilloscope", {})
        if "enabled" in scope:
            self.scope_enabled_check.setChecked(scope["enabled"])
        if "model" in scope:
            self.scope_model_combo.setCurrentText(scope["model"])
        if "resolution" in scope:
            self.scope_resolution_combo.setCurrentText(scope["resolution"])
        if "sampling_value" in scope:
            self.scope_sampling_value_spin.setValue(scope["sampling_value"])
        if "sampling_unit" in scope:
            self.scope_sampling_unit_combo.setCurrentText(scope["sampling_unit"])
        if "range_a" in scope:
            self.scope_range_a_combo.setCurrentText(scope["range_a"])
        if "range_b" in scope:
            self.scope_range_b_combo.setCurrentText(scope["range_b"])
        if "conv_factor_vref" in scope:
            self.scope_conv_factor_vref_spin.setValue(scope["conv_factor_vref"])

        analysis = data.get("online_analysis", {})
        if "decimation_enabled" in analysis:
            self.decimation_enabled_check.setChecked(analysis["decimation_enabled"])
        if "filter_cutoff" in analysis:
            self.filter_cutoff_spin.setValue(analysis["filter_cutoff"])
        if "filter_order" in analysis:
            self.filter_order_spin.setValue(analysis["filter_order"])
        if "resampling_frequency" in analysis:
            self.resampling_freq_edit.setText(analysis["resampling_frequency"])
        # window_size/buffer_duration: no longer restored -- both are now
        # auto-computed (see _compute_analysis_window_and_buffer). A
        # settings file saved before this change may still have these
        # keys; harmlessly ignored.

    def _on_save_settings(self):
        start_dir = self._settings.value("paths/last_settings_dir", "")
        default_name = str(Path(start_dir) / f"{self.experiment_name_edit.text()}_settings.json") if start_dir else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment settings", default_name, "JSON files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self._settings.setValue("paths/last_settings_dir", str(Path(path).parent))
        try:
            with open(path, "w") as f:
                json.dump(self._gather_settings_for_save(), f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save settings", f"{type(exc).__name__}: {exc}")
            return
        self.preview_text.appendPlainText(f"\nSaved experiment settings to {path}")

    def _on_load_settings(self):
        start_dir = self._settings.value("paths/last_settings_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Load experiment settings", start_dir, "JSON files (*.json)")
        if not path:
            return
        self._settings.setValue("paths/last_settings_dir", str(Path(path).parent))
        try:
            with open(path) as f:
                data = json.load(f)
            # "deistools_v2_experiment_settings" is the tag of files saved before the GUI moved into elma
            if data.get("type") not in ("elma_experiment_settings", "deistools_v2_experiment_settings"):
                raise ValueError("This doesn't look like an elma experiment settings file.")
            self._apply_loaded_settings(data)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load settings", f"{type(exc).__name__}: {exc}")
            return
        self.preview_text.appendPlainText(f"\nLoaded experiment settings from {path}")

    # ------------------------------------------------------------------ #
    # Actions -- preview / run
    # ------------------------------------------------------------------ #

    def _sampling_time_seconds(self) -> float:
        value = self.scope_sampling_value_spin.value()
        unit = self.scope_sampling_unit_combo.currentText()
        return value * _TIME_UNIT_TO_SECONDS[unit]

    def _compute_analysis_window_and_buffer(self, sampling_time_s: float):
        """(window_size_samples, buffer_duration_seconds) of the online analysis, or (None, None) if no
        multisine is loaded -- see elma.config.compute_analysis_window_and_buffer."""
        return C.compute_analysis_window_and_buffer(self._active_frequencies(), self._sequence_specs, sampling_time_s)

    def _update_analysis_auto_label(self):
        if not hasattr(self, "analysis_auto_label"):
            return  # still mid-construction; _build_ui() wires this up before it can fire
        try:
            sampling_time_s = self._sampling_time_seconds()
        except Exception:
            self.analysis_auto_label.setText("Analysis window / buffer duration: set a valid oscilloscope sampling interval.")
            return
        window_size, buffer_duration = self._compute_analysis_window_and_buffer(sampling_time_s)
        if window_size is None:
            self.analysis_auto_label.setText(
                "Analysis window / buffer duration: load a multisine design to compute "
                "(auto: window = 1x the period of the lowest active frequency, "
                "also roughly how often a new result appears)."
            )
            return
        self.analysis_auto_label.setText(
            f"Analysis window (auto): {window_size:,} samples "
            f"(~{window_size * sampling_time_s:.4g} s -- a new result appears about this often). "
            f"Buffer duration (auto): {buffer_duration:.4g} s -- covers the whole planned run "
            "if the sequence has a timed step, else 3x the period as a placeholder. "
            "(Window = 1x the period of the lowest active frequency; capture size below is 3x this window.)"
        )

    def _compute_capture_size(self, decimation_enabled: bool, sampling_time_s: float):
        """Oscilloscope capture size in samples, or None -- see elma.config.compute_capture_size."""
        return C.compute_capture_size(self._active_frequencies(), self._sequence_specs, decimation_enabled, sampling_time_s)

    def _resolve_awg_amplitude_volts(self):
        """(amplitude_volts, error_message) -- see elma.config.resolve_awg_amplitude_volts."""
        return C.resolve_awg_amplitude_volts(
            self._sequence_specs, self.awg_amplitude_type_combo.currentText(), self.awg_amplitude_spin.value()
        )

    def _resolve_scope_conv_factor(self):
        """(conv_factor_A_per_V, error_message) -- see elma.config.resolve_scope_conv_factor."""
        return C.resolve_scope_conv_factor(self._sequence_specs, self.scope_conv_factor_vref_spin.value())

    def _gather_configuration(self) -> dict:
        """The resolved configuration of the experiment as it is set up now (what Start Experiment
        builds the run from): the raw settings the widgets hold, resolved by elma.config for the
        loaded multisine design."""
        return C.resolve_configuration(
            self._gather_settings_for_save(), self._loaded_multisine, self._loaded_multisine_split
        )

    def _on_preview_configuration(self):
        config = self._gather_configuration()
        self.preview_text.setPlainText(json.dumps(config, indent=2))

    def _on_start_experiment(self):
        if not self._sequence_specs:
            QMessageBox.critical(self, "Cannot start", "Add at least one technique step to the sequence.")
            return
        if self.awg_enabled_check.isChecked() and not self._multisine_loaded():
            QMessageBox.critical(self, "Cannot start", "AWG is enabled but no multisine design is loaded.")
            return
        if self.awg_enabled_check.isChecked():
            _, amplitude_error = self._resolve_awg_amplitude_volts()
            if amplitude_error:
                QMessageBox.critical(self, "Cannot start", amplitude_error)
                return
        scope_enabled = self.scope_enabled_check.isChecked()
        # None of the scope-sizing/I-range checks below mean anything when
        # the scope itself is off (see _on_scope_enabled_toggled -- it also
        # forces decimation off in that case, so decimation_enabled is
        # always False here whenever scope_enabled is False).
        if scope_enabled:
            _, conv_factor_error = self._resolve_scope_conv_factor()
            if conv_factor_error:
                QMessageBox.critical(self, "Cannot start", conv_factor_error)
                return
        decimation_enabled = scope_enabled and self.decimation_enabled_check.isChecked()
        if decimation_enabled and not self._multisine_loaded():
            QMessageBox.critical(
                self, "Cannot start",
                "Online decimation/FFT-EIS is enabled but no multisine design is loaded "
                "(needed to know the lowest frequency and size the capture).",
            )
            return
        if scope_enabled and not decimation_enabled and not any(
            spec["type"] in _DURATION_STEP_TYPES for spec in self._sequence_specs
        ):
            QMessageBox.critical(
                self, "Cannot start",
                "Online decimation is disabled, so the raw capture size is computed from "
                "the sequence's timed step duration(s) -- add at least one CA/CALim/CP/CPLim/"
                "OCV/PEIS/GEIS step, or turn the oscilloscope off entirely if you don't need it.",
            )
            return

        # Fresh plots for this run: reset the accumulated potential/current
        # history and rebuild the figure with a 3rd (Nyquist) panel only
        # when online decimation/FFT-EIS is active for this run.
        self._plot_lines_read = 0
        self._plot_time = np.array([])
        self._plot_potential = np.array([])
        self._plot_current = np.array([])
        self._plot_measurement_file = (
            Path(self.saving_dir_edit.text()) / self.experiment_name_edit.text() / "measurement_data.txt"
        )
        self._build_plot_axes(with_impedance=decimation_enabled)
        self._peis_live_rows_shown = 0
        self._peis_geis_result = None
        self.peis_geis_save_btn.setEnabled(False)
        self._build_peis_geis_axes()
        self.peis_geis_results_text.clear()
        self.peis_geis_status_label.setText("Run in progress -- PEIS/GEIS results (if any) will appear here.")

        self.preview_text.setPlainText("Connecting to hardware...\n")
        try:
            self._deischannel = self._build_run()
            self._deischannel.start()
        except Exception as exc:
            self.preview_text.appendPlainText(f"\nFAILED: {type(exc).__name__}: {exc}")
            QMessageBox.critical(
                self,
                "Could not start experiment",
                f"{type(exc).__name__}: {exc}\n\n"
                "This is expected if no potentiostat/AWG/oscilloscope is actually "
                "connected -- this tab is a skeleton for wiring real hardware, not a simulator.",
            )
            self._deischannel = None
            return

        self.preview_text.appendPlainText("Experiment started.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._run_status_timer.start()

    def _on_stop_experiment(self):
        self._run_status_timer.stop()
        if self._deischannel is not None:
            deischannel = self._deischannel
            deischannel.stop()
            self._update_peis_geis_live_plot()
            self._save_peis_geis_run_results()
            self.preview_text.appendPlainText("\nStop requested.")
            # elma only saves the final online results when a technique ends on its
            # own, not on Stop -- but BlockCalculator(save_dir=...) already
            # wrote everything computed so far after each block, so nothing
            # more needs saving here.
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_check_run_finished(self):
        """
        Polled every second while a run is active: notices when it has
        ended BY ITSELF (sequence/duration completed, a hardware or
        software limit ended the last step, ...) rather than via the Stop
        button, and resets the Start/Stop buttons accordingly -- otherwise
        Stop stays enabled forever after a run that finished on its own.
        """
        if self._deischannel is None:
            self._run_status_timer.stop()
            return
        if self.live_plot_check.isChecked():
            self._on_update_plots()
        # PEIS/GEIS results have their own tab and are cheap to draw (a
        # point per frequency), so they update whether or not the
        # Potential/Current "Live plot" box is ticked.
        self._update_peis_geis_live_plot()
        if has_finished(self._deischannel):
            self._run_status_timer.stop()
            self._update_peis_geis_live_plot()  # points that arrived after the last tick
            self._save_peis_geis_run_results()
            self.preview_text.appendPlainText("\nExperiment finished.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _on_update_plots(self):
        """
        Redraws the embedded Potential(t)/Current(t) plots from
        measurement_data.txt (the same file + column layout pyeclab's own
        LivePlot reads, incrementally, so this works identically regardless
        of decimation mode), plus a live Nyquist plot of the latest
        per-block online impedance when online decimation is active for
        this run -- deliberately just the high-frequency, per-block
        FFT-EIS impedance already computed online; the low-frequency
        region (needs several periods for accuracy) isn't part of this
        online pipeline and is left for offline analysis.
        """
        if self._plot_measurement_file is not None and self._plot_measurement_file.exists():
            try:
                import pandas as pd

                new_rows = pd.read_csv(
                    self._plot_measurement_file, delimiter="\t", skiprows=self._plot_lines_read,
                )
                self._plot_lines_read += len(new_rows.index)
                if not new_rows.empty:
                    self._plot_time = np.append(self._plot_time, new_rows.iloc[:, 0].to_numpy())
                    self._plot_potential = np.append(self._plot_potential, new_rows.iloc[:, 1].to_numpy())
                    self._plot_current = np.append(self._plot_current, new_rows.iloc[:, 2].to_numpy() * 1000)
                    self._potential_line.set_data(self._plot_time, self._plot_potential)
                    self._current_line.set_data(self._plot_time, self._plot_current)
                    self._ax_potential.relim()
                    self._ax_potential.autoscale_view()
                    self._ax_current.relim()
                    self._ax_current.autoscale_view()
            except Exception:
                pass  # file header-only / mid-write / transient read race -- try again next tick

        if self._ax_impedance is not None:
            block_calculator = getattr(getattr(self._deischannel, "pico", None), "block_calculator", None)
            if block_calculator is not None and block_calculator.impedance_index > 0:
                latest = block_calculator.impedance[:, block_calculator.impedance_index - 1]
                self._impedance_line.set_data(latest.real, -latest.imag)
                self._ax_impedance.relim()
                self._ax_impedance.autoscale_view()

        self.canvas.draw_idle()

    def _peis_geis_channel_points(self):
        """(freqs, z, step_index) of the PEIS/GEIS points received so far, or None (thread-safe
        snapshot, see elma.peis_geis.PEISAwareChannel.peis_geis_points)."""
        channel_obj = getattr(self._deischannel, "potentiostat", None) if self._deischannel is not None else None
        points = getattr(channel_obj, "peis_geis_points", None)
        return points() if points is not None else None

    def _update_peis_geis_live_plot(self):
        """
        Live-updates the PEIS/GEIS tab while a sequence run's PEIS/GEIS step
        is active -- reads the points PEISAwareChannel has collected from
        Channel's own polling thread. Deliberately not a second poller on
        the device: reading the same channel handle from two threads at
        once is the kind of concurrent access that segfaulted the process
        during PEIS/GEIS hardware testing.
        """
        pts = self._peis_geis_channel_points()
        if pts is None:
            return
        freqs, z, step_index = pts
        if freqs.size == self._peis_live_rows_shown:
            return  # nothing new since the last tick
        self._peis_live_rows_shown = freqs.size
        self._draw_peis_geis_points(freqs, z, step_index)
        self.peis_geis_status_label.setText(f"Live from current run -- {freqs.size} frequency point(s) so far.")

    def _draw_peis_geis_points(self, freqs, z, step_index):
        """Redraws the PEIS/GEIS tab's Nyquist/Bode plots and results table.
        One trace per sequence step (a sequence can hold several PEIS/GEIS
        steps), each sorted by frequency so the Nyquist line connects
        neighbors regardless of sweep direction."""
        # The axes are rebuilt for every new point; carry over a zoom/pan the
        # user set (autoscaling is off for an axes they zoomed or panned).
        kept_views = [
            (ax.get_autoscalex_on(), ax.get_autoscaley_on(), ax.get_xlim(), ax.get_ylim())
            for ax in (self._ax_peis_nyquist, self._ax_peis_bode)
        ]
        self._build_peis_geis_axes()
        steps = np.unique(step_index)
        lines = [f"{'step':>5} {'freq (Hz)':>12} {'|Z| (ohm)':>12} {'phase (deg)':>12}"]
        for i, step in enumerate(steps):
            sel = step_index == step
            order = np.argsort(freqs[sel])
            f_s, z_s = freqs[sel][order], z[sel][order]
            color = f"C{i % 10}"
            label = f"step {step + 1}"
            self._ax_peis_nyquist.plot(z_s.real, -z_s.imag, "o-", color=color, label=label)
            self._ax_peis_bode.plot(f_s, np.abs(z_s), "o-", color=color, label=label)
            for f, zz in zip(f_s, z_s):
                lines.append(f"{step + 1:5d} {f:12.4g} {abs(zz):12.4g} {np.degrees(np.angle(zz)):12.3f}")
        self._ax_peis_nyquist.set_aspect("equal", adjustable="datalim")
        if steps.size > 1:
            self._ax_peis_bode.legend(fontsize=8)
        for ax, (auto_x, auto_y, xlim, ylim) in zip((self._ax_peis_nyquist, self._ax_peis_bode), kept_views):
            if not auto_x:
                ax.set_xlim(xlim)
            if not auto_y:
                ax.set_ylim(ylim)
        self.peis_geis_figure.tight_layout()
        self.peis_geis_toolbar.update()  # drop zoom history that points at the old axes
        self.peis_geis_canvas.draw_idle()
        self.peis_geis_results_text.setPlainText("\n".join(lines))

    def _build_run(self):
        """Connect to the instruments and build the run for the current configuration
        (elma.builder.build_run): a DEISchannel, or a RawCaptureRun / PotentiostatOnlyRun when online
        analysis / the oscilloscope is switched off. All expose start()/stop()."""
        return build_run(self._gather_configuration(), self._loaded_multisine, self._loaded_multisine_split)
