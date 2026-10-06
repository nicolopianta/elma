"""
Inspection tab: load a completed DEIS measurement and run offline analyses
on it.

Layout: a fixed-width left column holds controls (load + one group per
analysis), the right side stacks one widget per analysis -- more analyses
get appended there over time, each independent of the others.

First analysis: a slider plot. Potential(t)/Current(t) are stacked on the
left of the embedded figure (decimated for display, same approach as the
Multisine Designer's own plots); a Nyquist plot on the right shows
impedance.npy's block at the slider's current index, with a red marker on
the time-domain plots showing roughly where that block falls in the run.

Second control: DMFA on the low band. The online per-block impedance
(impedance.npy) already covers the whole frequency range, computed via
plain per-window FFT-EIS, on its own coarse per-block time axis (one
point roughly every window period) -- DMFA re-estimates just the low-band
frequencies from the saved decimated voltage/current (which only ever
contains low-band content in the first place, see
elma.design.waveform's decimation discussion) at its own, finer
time resolution (one point roughly every 20s). DMFA also extracts the
zero-frequency (DC) component, which the online pipeline has no
equivalent of at all -- deistools's run_dmfa() always computes it (a
separate Fermi-Dirac low-pass around f=0, applied to voltage and current
directly, since DC is resistive rather than an impedance to search for).

Rather than collapsing DMFA's own time resolution into a single
time-averaged value per frequency (which is what an earlier version of
this did, and is wrong -- DMFA's whole point is that it resolves how
these frequencies evolve over the run), running DMFA rebuilds the display
arrays (_frequencies/_impedance/_display_times) wholesale, on DMFA's own
time axis: the zero-frequency and low-band rows carry DMFA's genuine
per-point values, and every other (untouched, online-only) row is
zero-order-hold resampled from its own coarser block axis onto that same
grid, so the whole array shares one consistent time axis and the slider
shows every row changing honestly at its own native resolution instead of
silently flatlining the ones DMFA just computed. Always rebuilt from
_online_frequencies/_online_impedance (the as-loaded, untouched
originals) rather than from a previous DMFA result, so re-running with
different parameters doesn't compound resampling error.
"""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from deistools.processing import detrending
from deistools.processing.data_loader import find_technique_folders, load_experiment
from deistools.processing.dmfa_pipeline import run_dmfa_on_record, zero_order_hold_resample
from deistools.processing.spectrum import amplitude_spectrum, log_binned_max

from elma.gui.help_button import HelpButton, with_help
from elma.gui.plot_decimation import decimate_min_max
from elma.gui.plot_toolbar import canvas_with_toolbar


class InspectionTab(QWidget):
    """Load a DEIS measurement folder and run offline analyses on it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_dir = None
        self._time = None
        self._potential = None
        self._current = None
        self._impedance = None  # (n_freq, n_points) complex -- what the slider plot shows
        self._frequencies = None  # (n_freq,), matches _impedance's rows
        self._display_times = None  # (n_points,) seconds, matches _impedance's columns
        self._online_frequencies = None  # as loaded from metadata, never mutated
        self._online_impedance = None  # as loaded from impedance.npy, never mutated
        self._online_block_times = None  # approximate center time of each online block
        self._zero_voltage = None  # (n_points,) V, DMFA's zero-frequency-filtered potential(t)
        self._zero_current = None  # (n_points,) mA, DMFA's zero-frequency-filtered current(t)
        self._decimated_voltage = None
        self._decimated_current = None
        self._detrend_view = None  # last detrend result shown in the detrend/spectrum plot
        self._settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "elma", "Inspection")
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        outer = QHBoxLayout(self)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self._build_load_group())
        left_layout.addWidget(self._build_dmfa_group())
        left_layout.addStretch(1)
        left_scroll.setWidget(left_widget)
        outer.addWidget(left_scroll)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        self._right_layout = QVBoxLayout(right_widget)
        self._right_layout.addWidget(self._build_slider_plot_group())
        self._right_layout.addWidget(self._build_detrend_plot_group())
        self._right_layout.addStretch(1)
        right_scroll.setWidget(right_widget)
        outer.addWidget(right_scroll, 1)

    # ------------------------------------------------------------------ #
    # Left column -- load
    # ------------------------------------------------------------------ #

    def _build_load_group(self) -> QWidget:
        group = QGroupBox("Load DEIS data")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        load_btn = QPushButton("Load data folder...")
        load_btn.clicked.connect(self._on_load_data)
        row.addWidget(load_btn)
        row.addWidget(HelpButton(
            "Pick the experiment's saving folder (saving_directory/experiment_name "
            "from the Experiment Builder tab). Reads measurement_data.txt "
            "(potential/current vs time), metadata_deis_exp.json (frequency list), "
            "and the online results -- impedance (per-block FFT-EIS) plus the "
            "decimated voltage/current needed for DMFA below -- from "
            "pico_aquisition/cycle_*_sequence_*/ (concatenated in order when the "
            "sequence had several techniques; folders from before this layout, "
            "with the arrays in the experiment folder itself, still load). Missing "
            "files are skipped, not fatal, and shown as such in the status below."
        ))
        layout.addLayout(row)
        self.load_status_label = QLabel("No data loaded.")
        self.load_status_label.setStyleSheet("color: gray;")
        self.load_status_label.setWordWrap(True)
        layout.addWidget(self.load_status_label)
        return group

    def _on_load_data(self):
        start_dir = self._settings.value("paths/last_data_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select experiment data folder", start_dir)
        if not folder:
            return
        self._settings.setValue("paths/last_data_dir", folder)
        self._load_from_folder(Path(folder))

    @staticmethod
    def _technique_folders(folder: Path):
        """pico_aquisition/cycle_<loop>_sequence_<technique>/ folders in run order (see
        deistools.processing.data_loader.find_technique_folders)."""
        return find_technique_folders(folder)

    def _load_from_folder(self, folder: Path):
        """Reads the experiment folder with deistools.processing.data_loader.load_experiment; the
        notes it returns (what was loaded / missing / failed) become the status text."""
        self._data_dir = folder
        data = load_experiment(folder)
        status_parts = list(data.notes)
        self._time = data.time
        self._potential = data.potential
        self._current = None if data.current is None else data.current * 1000  # A -> mA
        self._frequencies = data.frequencies
        self._impedance = data.impedance
        self._decimated_voltage = data.decimated_voltage
        self._decimated_current = data.decimated_current
        if self._frequencies is not None and self._impedance is not None and self._frequencies.size != self._impedance.shape[0]:
            status_parts.append(
                "  WARNING: DMFA will be disabled until the frequency count and the impedance rows agree."
            )

        # Kept aside, untouched, as the source DMFA always rebuilds the
        # display arrays from (see _on_run_dmfa) -- re-running DMFA (e.g.
        # with different parameters) starts fresh from these rather than
        # resampling an already-resampled previous result.
        self._online_frequencies = None if self._frequencies is None else self._frequencies.copy()
        self._online_impedance = None if self._impedance is None else self._impedance.copy()
        self._online_block_times = None
        if (
            self._online_impedance is not None and self._online_impedance.shape[1] > 0
            and self._time is not None and self._time.size
        ):
            n_blocks = self._online_impedance.shape[1]
            total_duration = float(self._time.max())
            # Approximate: blocks are non-overlapping and roughly evenly
            # spaced across the run, so each block's own center time is
            # estimated from its position among n_blocks -- exact block
            # boundaries aren't saved by the online pipeline.
            self._online_block_times = total_duration * (np.arange(n_blocks) + 0.5) / n_blocks
        self._display_times = self._online_block_times
        self._zero_voltage = self._zero_current = None

        self.dmfa_status_label.setText("Low frequency: not computed.")
        self.load_status_label.setText(f"Loaded from {folder}:\n" + "\n".join(status_parts))
        self._refresh_slider_plot()

    # ------------------------------------------------------------------ #
    # Left column -- DMFA (low frequency)
    # ------------------------------------------------------------------ #

    def _build_dmfa_group(self) -> QWidget:
        group = QGroupBox("DMFA (low frequency)")
        form = QFormLayout(group)
        self.dmfa_cutoff_spin = QDoubleSpinBox()
        self.dmfa_cutoff_spin.setRange(1e-9, 1e9)
        self.dmfa_cutoff_spin.setDecimals(6)
        self.dmfa_cutoff_spin.setValue(1.0)
        form.addRow("Low-band cutoff (Hz)", with_help(
            self.dmfa_cutoff_spin,
            "Loaded frequencies below this are re-estimated via DMFA from the "
            "decimated voltage/current; the rest keep their original online "
            "per-block FFT-EIS values untouched. The decimated data only ever "
            "contains low-band content to begin with (the online lowpass filter "
            "removes everything else before decimation), so this should match "
            "wherever your design's split/low-band boundary actually was."
        ))
        self.dmfa_resampling_spin = QDoubleSpinBox()
        self.dmfa_resampling_spin.setRange(1e-6, 1e6)
        self.dmfa_resampling_spin.setDecimals(3)
        self.dmfa_resampling_spin.setValue(50.0)
        form.addRow("Resampling frequency (Hz)", with_help(
            self.dmfa_resampling_spin,
            "The sample rate the decimated voltage/current were actually saved "
            "at -- the 'Resampling frequency' value the run was configured with "
            "in the Experiment Builder tab's Online decimation / FFT-EIS section."
        ))

        self.dmfa_zero_bw_mult_spin = QDoubleSpinBox()
        self.dmfa_zero_bw_mult_spin.setRange(1e-6, 100.0)
        self.dmfa_zero_bw_mult_spin.setDecimals(4)
        self.dmfa_zero_bw_mult_spin.setSingleStep(0.1)
        self.dmfa_zero_bw_mult_spin.setValue(0.8)
        form.addRow("Zero-freq filter bandwidth (x lowest freq)", with_help(
            self.dmfa_zero_bw_mult_spin,
            "Fermi-Dirac low-pass bandwidth for the zero-frequency (DC) voltage/"
            "current extraction, as a multiple of the lowest loaded frequency "
            "(below the cutoff). Wider lets more of the lowest frequency's own "
            "content leak into the DC estimate; narrower rejects it more but "
            "responds to DC changes more slowly. Default 0.8x."
        ))
        self.dmfa_zero_order_spin = QSpinBox()
        self.dmfa_zero_order_spin.setRange(1, 10000)
        self.dmfa_zero_order_spin.setValue(100)
        form.addRow("Zero-freq filter order (n)", with_help(
            self.dmfa_zero_order_spin,
            "Fermi-Dirac filter steepness for the zero-frequency extraction -- "
            "higher is a sharper roll-off (closer to a brick-wall cutoff at the "
            "bandwidth above), lower is a gentler slope. Default 100."
        ))
        self.dmfa_multisine_bw_mult_spin = QDoubleSpinBox()
        self.dmfa_multisine_bw_mult_spin.setRange(1e-6, 100.0)
        self.dmfa_multisine_bw_mult_spin.setDecimals(4)
        self.dmfa_multisine_bw_mult_spin.setSingleStep(0.01)
        self.dmfa_multisine_bw_mult_spin.setValue(0.1)
        form.addRow("Multisine filter bandwidth (x lowest freq)", with_help(
            self.dmfa_multisine_bw_mult_spin,
            "Fermi-Dirac bandwidth around each low-band frequency's own peak, "
            "as a multiple of the lowest loaded frequency (below the cutoff) -- "
            "same reference frequency for every point in the band, not each "
            "point's own spacing to its neighbors. Default 0.1x."
        ))
        self.dmfa_multisine_order_spin = QSpinBox()
        self.dmfa_multisine_order_spin.setRange(1, 10000)
        self.dmfa_multisine_order_spin.setValue(8)
        form.addRow("Multisine filter order (n)", with_help(
            self.dmfa_multisine_order_spin,
            "Fermi-Dirac filter steepness for each low-band frequency's "
            "extraction. Default 8."
        ))

        self.dmfa_detrend_combo = QComboBox()
        self.dmfa_detrend_combo.addItems(detrending.METHODS)
        self.dmfa_detrend_combo.setCurrentText(detrending.METHOD_SPLINE)
        form.addRow("Detrend (before the FFT)", with_help(
            self.dmfa_detrend_combo,
            "DMFA transforms the whole record at once, and the record is not periodic: "
            "the potential drifts between its first and last sample. That jump leaks into "
            "every bin -- including the lowest multisine tones -- and wrecks the first and "
            "last DMFA points. A slow trend is estimated and removed from the voltage and "
            "from the current first; it is added back to the zero-frequency (DC) result "
            "afterwards, since it is the physical DC part.\n\n"
            "None: nothing removed.\n"
            "Linear (end points): a line through the mean of the first and of the last "
            "period of the lowest frequency. Cheap; only fixes a straight drift.\n"
            "Polynomial (iterative): polynomial of the given order. Rarely a good fit for "
            "real drift (knees, relaxations).\n"
            "Smooth spline (iterative): cubic spline with the knot spacing below; follows "
            "curved drift and is the default.\n\n"
            "The two iterative methods fit block means over one period of the lowest "
            "frequency and re-estimate the tones to subtract them before refitting, so the "
            "trend cannot absorb the signal. (A mirrored extension of the record is "
            "deliberately not offered: it scrambles the tones' phase.)"
        ))
        self.dmfa_detrend_order_spin = QSpinBox()
        self.dmfa_detrend_order_spin.setRange(1, 20)
        self.dmfa_detrend_order_spin.setValue(3)
        form.addRow("Polynomial order", with_help(
            self.dmfa_detrend_order_spin, "Order of the polynomial trend (Polynomial method only)."
        ))
        self.dmfa_detrend_knots_spin = QDoubleSpinBox()
        self.dmfa_detrend_knots_spin.setRange(1.0, 1000.0)
        self.dmfa_detrend_knots_spin.setDecimals(1)
        self.dmfa_detrend_knots_spin.setValue(3.0)
        form.addRow("Spline knot spacing (x lowest period)", with_help(
            self.dmfa_detrend_knots_spin,
            "Distance between the spline's knots, in periods of the lowest frequency "
            "(Smooth spline method only). Smaller follows faster drift (needed near "
            "sharp changes such as the start of a step); larger is stiffer. Default 3."
        ))
        self.dmfa_detrend_combo.currentTextChanged.connect(self._on_dmfa_detrend_method_changed)
        self._on_dmfa_detrend_method_changed(self.dmfa_detrend_combo.currentText())

        run_btn = QPushButton("Perform DMFA (low frequency)")
        run_btn.clicked.connect(self._on_run_dmfa)
        form.addRow(run_btn)
        self.dmfa_status_label = QLabel("Low frequency: not computed.")
        self.dmfa_status_label.setStyleSheet("color: gray;")
        self.dmfa_status_label.setWordWrap(True)
        form.addRow(self.dmfa_status_label)
        return group

    def _detrend_decimated(self, low_freqs, fs, n):
        """Applies the Detrend settings to the first n decimated samples of voltage and
        current. Returns (method, voltage_detrended, voltage_trend, current_detrended,
        current_trend)."""
        f_min = float(np.min(low_freqs))
        method = self.dmfa_detrend_combo.currentText()
        kwargs = dict(
            order=self.dmfa_detrend_order_spin.value(),
            knot_spacing_periods=self.dmfa_detrend_knots_spin.value(),
            half_bw=self.dmfa_multisine_bw_mult_spin.value() * f_min,
        )
        voltage_det, voltage_trend = detrending.detrend(self._decimated_voltage[:n], fs, low_freqs, method, **kwargs)
        current_det, current_trend = detrending.detrend(self._decimated_current[:n], fs, low_freqs, method, **kwargs)
        return method, voltage_det, voltage_trend, current_det, current_trend

    def _store_detrend_view(self, low_freqs, fs, n, method, voltage_det, voltage_trend, current_det, current_trend):
        self._detrend_view = dict(
            freqs=np.asarray(low_freqs, dtype=float), fs=fs, n=n, method=method,
            raw_v=self._decimated_voltage[:n], raw_i=self._decimated_current[:n],
            v_trend=voltage_trend, i_trend=current_trend, v_det=voltage_det, i_det=current_det,
        )

    def _build_detrend_plot_group(self) -> QWidget:
        group = QGroupBox("Decimated data: trend, detrending and spectrum")
        layout = QVBoxLayout(group)
        top = QHBoxLayout()
        update_btn = QPushButton("Update (uses the Detrend settings on the left)")
        update_btn.clicked.connect(self._on_update_detrend_plot)
        top.addWidget(update_btn)
        top.addWidget(HelpButton(
            "Shows the decimated potential and current that DMFA works on, per column "
            "(potential left, current right):\n"
            "- top: the data and the trend that the chosen Detrend method estimated;\n"
            "- middle: the data minus the trend, which is what gets Fourier-transformed;\n"
            "- bottom: single-sided amplitude spectrum before and after detrending, with "
            "the low-band multisine frequencies marked. Detrending should lower the "
            "broad 1/f skirt of the drift between the tones, not the tones themselves.\n\n"
            "It is refreshed automatically whenever DMFA is run; this button recomputes "
            "it without running DMFA, e.g. to compare methods quickly. The 'Low-band "
            "cutoff', 'Resampling frequency' and filter bandwidth settings on the left "
            "are used, as for DMFA."
        ))
        top.addStretch(1)
        layout.addLayout(top)
        self.detrend_status_label = QLabel("Load data and press Update, or run DMFA.")
        self.detrend_status_label.setStyleSheet("color: gray;")
        self.detrend_status_label.setWordWrap(True)
        layout.addWidget(self.detrend_status_label)

        self._detrend_figure = Figure(figsize=(10, 9))
        self._detrend_canvas = FigureCanvas(self._detrend_figure)
        plot_widget, self._detrend_toolbar = canvas_with_toolbar(self._detrend_canvas)
        plot_widget.setMinimumHeight(720)
        layout.addWidget(plot_widget)
        self._refresh_detrend_plot()
        return group

    def _on_update_detrend_plot(self):
        if self._decimated_voltage is None or self._decimated_current is None:
            QMessageBox.critical(self, "Cannot show spectrum", "No decimated voltage/current data loaded.")
            return
        if self._online_frequencies is None:
            QMessageBox.critical(self, "Cannot show spectrum", "No frequency list loaded.")
            return
        cutoff = self.dmfa_cutoff_spin.value()
        low_freqs = self._online_frequencies[self._online_frequencies < cutoff]
        if low_freqs.size == 0:
            QMessageBox.critical(self, "Cannot show spectrum", f"No loaded frequencies below {cutoff:g} Hz.")
            return
        fs = self.dmfa_resampling_spin.value()
        n = min(self._decimated_voltage.size, self._decimated_current.size)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            method, v_det, v_trend, i_det, i_trend = self._detrend_decimated(low_freqs, fs, n)
            self._store_detrend_view(low_freqs, fs, n, method, v_det, v_trend, i_det, i_trend)
            self._refresh_detrend_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Detrend failed", f"{type(exc).__name__}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _refresh_detrend_plot(self):
        fig = self._detrend_figure
        fig.clear()
        view = self._detrend_view
        if view is None:
            fig.text(0.5, 0.5, "No detrend result yet", ha="center", va="center", color="gray")
            self._detrend_canvas.draw_idle()
            if hasattr(self, "_detrend_toolbar"):
                self._detrend_toolbar.update()
            return

        fs, n, method = view["fs"], view["n"], view["method"]
        detrended = method != detrending.METHOD_NONE
        t = np.arange(n) / fs
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.4])
        columns = (
            ("Potential / V", 1.0, view["raw_v"], view["v_trend"], view["v_det"]),
            ("Current / mA", 1000.0, view["raw_i"], view["i_trend"], view["i_det"]),
        )
        ptp_text = []
        for c, (label, scale, raw, trend, det) in enumerate(columns):
            raw = np.asarray(raw, dtype=float) * scale
            trend = np.asarray(trend, dtype=float) * scale
            det = np.asarray(det, dtype=float) * scale
            ax_data = fig.add_subplot(gs[0, c])
            ax_res = fig.add_subplot(gs[1, c], sharex=ax_data)
            ax_spec = fig.add_subplot(gs[2, c])

            t_plot, raw_plot = decimate_min_max(t, raw)
            ax_data.plot(t_plot, raw_plot, "-", color="C0", linewidth=0.7, label="decimated data")
            if detrended:
                step = max(1, n // 5000)
                ax_data.plot(t[::step], trend[::step], "-", color="tab:red", linewidth=1.3, label=f"trend ({method})")
            ax_data.set_ylabel(label)
            ax_data.set_title("Data and trend" if c == 0 else "")
            ax_data.grid(True)
            ax_data.legend(fontsize=7, loc="upper right")

            t_plot, det_plot = decimate_min_max(t, det)
            ax_res.plot(t_plot, det_plot, "-", color="C2", linewidth=0.7)
            ax_res.set_ylabel(label)
            ax_res.set_xlabel("Time / s")
            ax_res.set_title("Data minus trend (what is Fourier-transformed)" if detrended else "No detrending applied")
            ax_res.grid(True)

            f_raw, a_raw = amplitude_spectrum(raw, fs)
            x_plot, y_plot = log_binned_max(f_raw, a_raw)
            ax_spec.plot(x_plot, y_plot, "-", color="C0", linewidth=0.8, label="decimated data")
            if detrended:
                f_det, a_det = amplitude_spectrum(det, fs)
                x_plot, y_plot = log_binned_max(f_det, a_det)
                ax_spec.plot(x_plot, y_plot, "-", color="C2", linewidth=0.8, label="detrended")
            for f0 in view["freqs"]:
                ax_spec.axvline(f0, color="gray", linewidth=0.5, alpha=0.4)
            ax_spec.set_xscale("log")
            ax_spec.set_yscale("log")
            ax_spec.set_xlabel("Frequency / Hz")
            ax_spec.set_ylabel(label.replace("/", "(amplitude) /", 1))
            ax_spec.set_title("Amplitude spectrum (grey: low-band multisine frequencies)" if c == 0 else "")
            ax_spec.grid(True, which="both", alpha=0.4)
            ax_spec.legend(fontsize=7, loc="upper right")
            ptp_text.append(f"{label.split(' /')[0]}: trend spans {np.ptp(trend):.4g} {label.split('/ ')[1]}")

        fig.tight_layout()
        self._detrend_canvas.draw_idle()
        if hasattr(self, "_detrend_toolbar"):
            self._detrend_toolbar.update()  # a rebuilt figure invalidates the zoom history
        self.detrend_status_label.setText(
            f"{n:,} samples at {fs:g} Hz ({n / fs:.0f} s); detrend: {method}; "
            f"lowest tone {float(np.min(view['freqs'])):g} Hz; FFT bin spacing {fs / n:.3g} Hz. "
            + ("; ".join(ptp_text) + "." if detrended else "")
        )

    def _on_dmfa_detrend_method_changed(self, method: str):
        self.dmfa_detrend_order_spin.setEnabled(method == detrending.METHOD_POLY)
        self.dmfa_detrend_knots_spin.setEnabled(method == detrending.METHOD_SPLINE)

    def _on_run_dmfa(self):
        if self._decimated_voltage is None or self._decimated_current is None:
            QMessageBox.critical(self, "Cannot run DMFA", "No decimated voltage/current data loaded.")
            return
        if self._online_frequencies is None or self._online_impedance is None:
            QMessageBox.critical(self, "Cannot run DMFA", "No frequency list / impedance array loaded.")
            return
        if self._online_frequencies.size != self._online_impedance.shape[0]:
            QMessageBox.critical(
                self, "Cannot run DMFA",
                f"Frequency count ({self._online_frequencies.size}) doesn't match impedance.npy's "
                f"rows ({self._online_impedance.shape[0]}).",
            )
            return
        if self._online_block_times is None:
            QMessageBox.critical(
                self, "Cannot run DMFA",
                "Need measurement_data.txt's time column to line up DMFA's own time "
                "resolution against the online blocks' -- load a folder that has it.",
            )
            return

        # Always rebuilt from the untouched, as-loaded _online_frequencies/
        # _online_impedance -- a re-run (different cutoff/bandwidth/order)
        # starts fresh rather than resampling an already-resampled result.
        cutoff = self.dmfa_cutoff_spin.value()
        low_mask = self._online_frequencies < cutoff
        if not np.any(low_mask):
            QMessageBox.critical(self, "Cannot run DMFA", f"No loaded frequencies below {cutoff:g} Hz.")
            return
        low_freqs = self._online_frequencies[low_mask]

        resampling_frequency = self.dmfa_resampling_spin.value()
        f_min = float(np.min(low_freqs))
        bw_zero_mult = self.dmfa_zero_bw_mult_spin.value()
        bw_filter_mult = self.dmfa_multisine_bw_mult_spin.value()
        zero_filter_order = self.dmfa_zero_order_spin.value()
        filter_order = self.dmfa_multisine_order_spin.value()
        bw_zero = bw_zero_mult * f_min
        bw_filter = bw_filter_mult * f_min
        npts_elab_seconds = 20.0   # time resolution of one DMFA point

        try:
            result = run_dmfa_on_record(
                self._decimated_voltage, self._decimated_current, low_freqs, resampling_frequency,
                bw_zero_mult=bw_zero_mult, bw_filter_mult=bw_filter_mult,
                zero_filter_order=zero_filter_order, filter_order=filter_order,
                seconds_per_point=npts_elab_seconds,
                detrend_method=self.dmfa_detrend_combo.currentText(),
                detrend_order=self.dmfa_detrend_order_spin.value(),
                knot_spacing_periods=self.dmfa_detrend_knots_spin.value(),
                keep_trends=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "DMFA failed", f"{type(exc).__name__}: {exc}")
            return
        detrend_method = result.detrend_method
        npts_elab = result.impedance.shape[1]
        n = result.n_samples
        # what the detrend/spectrum section shows: the data, the trend, and data minus trend
        self._store_detrend_view(
            low_freqs, resampling_frequency, n, detrend_method,
            np.asarray(self._decimated_voltage[:n], dtype=float) - result.voltage_trend, result.voltage_trend,
            np.asarray(self._decimated_current[:n], dtype=float) - result.current_trend, result.current_trend,
        )

        dmfa_time = result.time
        dmfa_impedance = result.impedance
        # Zero-frequency (DC): voltage and current are extracted separately around f=0 rather than
        # giving an impedance directly -- V(t)/I(t) is the DC resistance's own time series, at the same
        # time resolution as dmfa_impedance.
        zero_resistance_series = (result.zero_voltage / result.zero_current).astype(complex)
        self._zero_voltage = np.asarray(result.zero_voltage, dtype=float)
        self._zero_current = np.asarray(result.zero_current, dtype=float) * 1000  # A -> mA, matches self._current's unit

        # The online per-block rows (everything NOT re-estimated by DMFA) only ever had n_online_blocks
        # values on their own, coarser (~window-period-spaced) time axis -- hold them onto DMFA's finer
        # time grid so every row of the display array shares one time axis, without inventing new
        # information for the rows DMFA didn't touch.
        high_resampled = zero_order_hold_resample(
            self._online_impedance[~low_mask, :], self._online_block_times, dmfa_time
        )

        self._frequencies = np.concatenate([[0.0], self._online_frequencies[low_mask], self._online_frequencies[~low_mask]])
        self._impedance = np.vstack([
            zero_resistance_series[None, :],
            dmfa_impedance,
            high_resampled,
        ])
        self._display_times = dmfa_time
        self._refresh_detrend_plot()

        self.dmfa_status_label.setText(
            f"Low frequency: computed ({low_freqs.size} freq(s) below {cutoff:g} Hz, "
            f"{npts_elab} time-resolved DMFA point(s) -- the slider now steps through these, "
            "not the online blocks). Other rows zero-order-hold resampled from the online "
            f"blocks onto this same time axis. Zero-frequency (DC) resistance ranges "
            f"{zero_resistance_series.real.min():.4g}-{zero_resistance_series.real.max():.4g} ohm "
            f"(bw_zero={bw_zero:.4g} Hz, n={zero_filter_order}; bw_filter={bw_filter:.4g} Hz, n={filter_order}; "
            f"detrend: {detrend_method})."
        )
        self._refresh_slider_plot()

    # ------------------------------------------------------------------ #
    # Right column -- slider plot (potential/current vs time + Nyquist vs block)
    # ------------------------------------------------------------------ #

    def _build_slider_plot_group(self) -> QWidget:
        group = QGroupBox("Time-resolved impedance (slider)")
        layout = QVBoxLayout(group)

        self._slider_figure = Figure(figsize=(9, 5))
        self._slider_canvas = FigureCanvas(self._slider_figure)
        self._slider_canvas.setMinimumHeight(420)
        slider_plot_widget, self._slider_toolbar = canvas_with_toolbar(self._slider_canvas)
        slider_plot_widget.setMinimumHeight(460)
        layout.addWidget(slider_plot_widget)

        slider_row = QHBoxLayout()
        self.block_slider = QSlider(Qt.Horizontal)
        self.block_slider.setMinimum(0)
        self.block_slider.setMaximum(0)
        self.block_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.block_slider, 1)
        self.block_slider_label = QLabel("No data loaded.")
        self.block_slider_label.setMinimumWidth(180)
        slider_row.addWidget(self.block_slider_label)
        layout.addLayout(slider_row)

        self._build_slider_axes()
        return group

    def _build_slider_axes(self):
        self._slider_figure.clear()
        gs = self._slider_figure.add_gridspec(2, 2, width_ratios=[1.3, 1])
        self._ax_potential = self._slider_figure.add_subplot(gs[0, 0])
        self._ax_current = self._slider_figure.add_subplot(gs[1, 0], sharex=self._ax_potential)
        self._ax_nyquist = self._slider_figure.add_subplot(gs[:, 1])

        self._ax_potential.set_ylabel("Potential / V")
        self._ax_potential.grid(True)
        self._ax_current.set_ylabel("Current / mA")
        self._ax_current.set_xlabel("Time / s")
        self._ax_current.grid(True)
        self._ax_nyquist.set_xlabel("Z' / Ohm")
        self._ax_nyquist.set_ylabel("-Z'' / Ohm")
        self._ax_nyquist.set_title("Block impedance")
        self._ax_nyquist.set_aspect("equal", adjustable="datalim")
        self._ax_nyquist.grid(True)

        (self._potential_line,) = self._ax_potential.plot([], [], "-", color="C0", linewidth=0.8, label="Potential")
        (self._current_line,) = self._ax_current.plot([], [], "-", color="C1", linewidth=0.8, label="Current")
        (self._zero_voltage_line,) = self._ax_potential.plot(
            [], [], "o-", color="tab:red", linewidth=1.2, markersize=3, label="DC (DMFA zero-freq)"
        )
        (self._zero_current_line,) = self._ax_current.plot(
            [], [], "o-", color="tab:red", linewidth=1.2, markersize=3, label="DC (DMFA zero-freq)"
        )
        self._potential_time_marker = self._ax_potential.axvline(0, color="red", linewidth=1, alpha=0.7)
        self._current_time_marker = self._ax_current.axvline(0, color="red", linewidth=1, alpha=0.7)
        (self._nyquist_line,) = self._ax_nyquist.plot([], [], "o-", color="C2")
        self._slider_figure.tight_layout()
        if hasattr(self, "_slider_toolbar"):
            self._slider_toolbar.update()  # a rebuilt figure invalidates the zoom history

    def _refresh_slider_plot(self):
        self._build_slider_axes()
        if self._time is not None and self._potential is not None and self._time.size:
            t_plot, v_plot = decimate_min_max(self._time, self._potential)
            self._potential_line.set_data(t_plot, v_plot)
        if self._time is not None and self._current is not None and self._time.size:
            t_plot, c_plot = decimate_min_max(self._time, self._current)
            self._current_line.set_data(t_plot, c_plot)
        if self._zero_voltage is not None and self._display_times is not None:
            self._zero_voltage_line.set_data(self._display_times, self._zero_voltage)
        if self._zero_current is not None and self._display_times is not None:
            self._zero_current_line.set_data(self._display_times, self._zero_current)
        if self._potential_line.get_xdata().size or self._zero_voltage_line.get_xdata().size:
            self._ax_potential.relim()
            self._ax_potential.autoscale_view()
            self._ax_potential.legend(fontsize=7, loc="upper right")
        if self._current_line.get_xdata().size or self._zero_current_line.get_xdata().size:
            self._ax_current.relim()
            self._ax_current.autoscale_view()
            self._ax_current.legend(fontsize=7, loc="upper right")

        n_blocks = 0 if self._impedance is None else self._impedance.shape[1]
        self.block_slider.setMaximum(max(0, n_blocks - 1))
        self._on_slider_changed(self.block_slider.value())

    def _on_slider_changed(self, index: int):
        if self._impedance is None or self._impedance.shape[1] == 0:
            self.block_slider_label.setText("No impedance blocks loaded.")
            self._slider_canvas.draw_idle()
            return
        n_blocks = self._impedance.shape[1]
        index = min(max(index, 0), n_blocks - 1)
        z = self._impedance[:, index]
        # The zero-frequency row (V_dc / I_dc, real by construction, sign follows the current direction)
        # is not an impedance: it is left out of the Nyquist plot. Its DC voltage/current are in the
        # potential/current panels.
        if self._frequencies is not None and self._frequencies.size == z.size:
            z = z[self._frequencies > 0]
        self._nyquist_line.set_data(z.real, -z.imag)
        self._ax_nyquist.relim()
        self._ax_nyquist.autoscale_view()

        if self._display_times is not None and self._display_times.size == n_blocks:
            t_center = float(self._display_times[index])
            self._potential_time_marker.set_xdata([t_center, t_center])
            self._current_time_marker.set_xdata([t_center, t_center])
            self.block_slider_label.setText(f"Point {index + 1}/{n_blocks}  (~{t_center:.4g} s)")
        else:
            self.block_slider_label.setText(f"Point {index + 1}/{n_blocks}")

        self._slider_canvas.draw_idle()
