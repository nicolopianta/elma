"""
PEIS (Potentio EIS) and GEIS (Galvano EIS) -- EC-Lab's own built-in
single-sine impedance techniques, available as steps of a technique
sequence (and in the elma GUI's Experiment Builder). They measure a cell's impedance through the
potentiostat's own internal sine generator and ADC (one integrated,
self-clocked instrument), which also makes them an independent cross-check
for the multisine-DEIS pipeline (external AWG + PicoScope, two instruments
with no shared clock): this is how the >30 kHz discrepancy in that
pipeline was isolated -- PEIS matched theory on the same cell where the
multisine measurement didn't.

pyeclab (the project's own EC-Lab wrapper) has no PEIS/GEIS technique
classes -- only CA/CALim/CP/CPLim/OCV. These two dataclasses follow the
exact same ECC_parm/make_ecc_parm/make_ecc_parms/choose_ecc_file/
make_technique pattern pyeclab's own ChronoAmperometry uses, so they plug
into the same sequence/LoadTechnique machinery. Parameter names and the set
of parameters each technique takes are from the official "EC-Lab
Development Package.pdf", section 7.11 (PEIS, technique ID 104) and 7.13
(GEIS, technique ID 107) -- notably PEIS takes no I_Range/E_Range/
Bandwidth at all (unlike CA), while GEIS does take I_Range (but not
E_Range/Bandwidth). Getting that parameter set wrong fails silently or
confusingly on real hardware, so it's copied verbatim from the documented
tables, not guessed.

The frequency-sweep results (process 1: freq, |Ewe|, |I|, Phase(Zwe), ...)
are decoded and collected by PEISAwareChannel, below. Phase comes back in RADIANS although nothing in the SDK docs
says so -- confirmed against a known theoretical dummy-cell impedance,
where the numbers only lined up after converting.

Duration_step is the length of the INITIAL HOLD at the step potential/
current (process 0) before the frequency sweep starts, not a timeout: no
impedance point can appear before it has elapsed.
"""
from dataclasses import dataclass, field

import time

import numpy as np
import pyeclab.api.kbio_types as KBIO
from pyeclab import Channel
from pyeclab.api.kbio_tech import ECC_parm, make_ecc_parm, make_ecc_parms
from pyeclab.device import BiologicDevice

# PEIS/GEIS technique IDs (EC-Lab Development Package doc, sections 7.11 / 7.13)
PEIS_GEIS_TECH_IDS = (104, 107)


@dataclass
class PEISTechnique:
    device: BiologicDevice
    vs_initial: bool
    initial_voltage_step: float
    duration_step: float
    record_every_dT: float
    record_every_dI: float
    initial_frequency: float
    final_frequency: float
    sweep_linear: bool
    amplitude_voltage: float
    frequency_number: int
    average_n_times: int
    correction: bool
    wait_for_steady: float
    xctr: int | None = None
    ecc_file: str | None = field(init=False, default=None)
    ecc_params: KBIO.EccParams | None = field(init=False, default=None)

    def make_params(self):
        names = {
            "vs_initial": ECC_parm("vs_initial", bool),
            "initial_voltage_step": ECC_parm("Initial_Voltage_step", float),
            "duration_step": ECC_parm("Duration_step", float),
            "record_every_dT": ECC_parm("Record_every_dT", float),
            "record_every_dI": ECC_parm("Record_every_dI", float),
            "final_frequency": ECC_parm("Final_frequency", float),
            "initial_frequency": ECC_parm("Initial_frequency", float),
            "sweep": ECC_parm("sweep", bool),
            "amplitude_voltage": ECC_parm("Amplitude_Voltage", float),
            "frequency_number": ECC_parm("Frequency_number", int),
            "average_n_times": ECC_parm("Average_N_times", int),
            "correction": ECC_parm("Correction", bool),
            "wait_for_steady": ECC_parm("Wait_for_steady", float),
            "xctr": ECC_parm("xctr", int),
        }
        params_list = [
            make_ecc_parm(self.device, names["vs_initial"], self.vs_initial),
            make_ecc_parm(self.device, names["initial_voltage_step"], self.initial_voltage_step),
            make_ecc_parm(self.device, names["duration_step"], self.duration_step),
            make_ecc_parm(self.device, names["record_every_dT"], self.record_every_dT),
            make_ecc_parm(self.device, names["record_every_dI"], self.record_every_dI),
            make_ecc_parm(self.device, names["final_frequency"], self.final_frequency),
            make_ecc_parm(self.device, names["initial_frequency"], self.initial_frequency),
            make_ecc_parm(self.device, names["sweep"], self.sweep_linear),
            make_ecc_parm(self.device, names["amplitude_voltage"], self.amplitude_voltage),
            make_ecc_parm(self.device, names["frequency_number"], self.frequency_number),
            make_ecc_parm(self.device, names["average_n_times"], self.average_n_times),
            make_ecc_parm(self.device, names["correction"], self.correction),
            make_ecc_parm(self.device, names["wait_for_steady"], self.wait_for_steady),
        ]
        if self.xctr:
            params_list.append(make_ecc_parm(self.device, names["xctr"], self.xctr))
        return make_ecc_parms(self.device, *params_list)

    def choose_ecc_file(self):
        return "peis.ecc" if self.device.is_VMP3 else "peis4.ecc"

    def make_technique(self):
        self.ecc_file = self.choose_ecc_file()
        self.ecc_params = self.make_params()


@dataclass
class GEISTechnique:
    device: BiologicDevice
    vs_initial: bool
    initial_current_step: float
    duration_step: float
    record_every_dT: float
    record_every_dE: float
    initial_frequency: float
    final_frequency: float
    sweep_linear: bool
    amplitude_current: float
    frequency_number: int
    average_n_times: int
    correction: bool
    wait_for_steady: float
    i_range: "KBIO.I_RANGE"
    xctr: int | None = None
    ecc_file: str | None = field(init=False, default=None)
    ecc_params: KBIO.EccParams | None = field(init=False, default=None)

    def make_params(self):
        names = {
            "vs_initial": ECC_parm("vs_initial", bool),
            "initial_current_step": ECC_parm("Initial_Current_step", float),
            "duration_step": ECC_parm("Duration_step", float),
            "record_every_dT": ECC_parm("Record_every_dT", float),
            "record_every_dE": ECC_parm("Record_every_dE", float),
            "final_frequency": ECC_parm("Final_frequency", float),
            "initial_frequency": ECC_parm("Initial_frequency", float),
            "sweep": ECC_parm("sweep", bool),
            "amplitude_current": ECC_parm("Amplitude_Current", float),
            "frequency_number": ECC_parm("Frequency_number", int),
            "average_n_times": ECC_parm("Average_N_times", int),
            "correction": ECC_parm("Correction", bool),
            "wait_for_steady": ECC_parm("Wait_for_steady", float),
            "i_range": ECC_parm("I_Range", int),
            "xctr": ECC_parm("xctr", int),
        }
        params_list = [
            make_ecc_parm(self.device, names["vs_initial"], self.vs_initial),
            make_ecc_parm(self.device, names["initial_current_step"], self.initial_current_step),
            make_ecc_parm(self.device, names["duration_step"], self.duration_step),
            make_ecc_parm(self.device, names["record_every_dT"], self.record_every_dT),
            make_ecc_parm(self.device, names["record_every_dE"], self.record_every_dE),
            make_ecc_parm(self.device, names["final_frequency"], self.final_frequency),
            make_ecc_parm(self.device, names["initial_frequency"], self.initial_frequency),
            make_ecc_parm(self.device, names["sweep"], self.sweep_linear),
            make_ecc_parm(self.device, names["amplitude_current"], self.amplitude_current),
            make_ecc_parm(self.device, names["frequency_number"], self.frequency_number),
            make_ecc_parm(self.device, names["average_n_times"], self.average_n_times),
            make_ecc_parm(self.device, names["correction"], self.correction),
            make_ecc_parm(self.device, names["wait_for_steady"], self.wait_for_steady),
            make_ecc_parm(self.device, names["i_range"], self.i_range.value),
        ]
        if self.xctr:
            params_list.append(make_ecc_parm(self.device, names["xctr"], self.xctr))
        return make_ecc_parms(self.device, *params_list)

    def choose_ecc_file(self):
        return "geis.ecc" if self.device.is_VMP3 else "geis4.ecc"

    def make_technique(self):
        self.ecc_file = self.choose_ecc_file()
        self.ecc_params = self.make_params()


class PEISAwareChannel(Channel):
    """
    pyeclab's Channel._get_converted_buffer() always calls buffer_converters.base(), which
    hardcodes the process-0 column layout (t_high, t_low, Ewe, I) -- right for CA/CP/OCV, which
    only ever emit process-0 data. PEIS/GEIS also emit process-1 buffers (the frequency-sweep
    result rows: freq, |Ewe|, |I|, phase, ...), and base() would misread THOSE columns as if
    they were (t, Ewe, I) too, writing garbage rows into measurement_data.txt and the live plots.

    Process-1 rows are decoded here instead (phase comes back in RADIANS) and appended to
    self.peis_freqs / self.peis_z, with the sequence step they belong to in
    self.peis_step_index, for a GUI to plot live and save; they contribute zero rows to the
    (t, Ewe, I) stream. Process-0 of a PEIS/GEIS step (the initial hold, sampled by EC-Lab at its
    ~24 us timebase -- ~680k rows per 100 s when Record_every_dT is 0, tens of MB of text) is
    dropped rather than written. Every other technique goes through the normal inherited path.

    Polling: process-1 buffers are transient -- in direct testing 0.3 s polling missed every
    result row while 0.02 s caught them all -- so _run (a copy of Channel._run, which sleeps a
    fixed 1 s) sleeps 0.02 s while a PEIS/GEIS technique is active and 1 s otherwise.
    Copy of Channel._run: keep it in step with the installed pyeclab version.
    """

    PEIS_GEIS_TECH_IDS = PEIS_GEIS_TECH_IDS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peis_freqs = []
        self.peis_z = []
        self.peis_step_index = []

    def _get_measurement_values(self):
        self._get_data()
        if self.data_info.TechniqueID in self.PEIS_GEIS_TECH_IDS:
            if self.data_info.ProcessIndex == 1 and self.data_info.NbRows > 0:
                cols = self.data_info.NbCols
                buf = self.data_buffer
                for r in range(self.data_info.NbRows):
                    row = [buf[r * cols + c] for c in range(cols)]
                    freq = self.bio_device.ConvertNumericIntoSingle(row[0])
                    abs_ewe = self.bio_device.ConvertNumericIntoSingle(row[1])
                    abs_i = self.bio_device.ConvertNumericIntoSingle(row[2])
                    phase_rad = self.bio_device.ConvertNumericIntoSingle(row[3])
                    z = (abs_ewe / abs_i) * np.exp(1j * phase_rad) if abs_i else complex("nan")
                    # step index last: readers on another thread treat len(peis_freqs) as the
                    # number of complete rows, so freq is appended after the other two.
                    self.peis_z.append(z)
                    self.peis_step_index.append(self.data_info.TechniqueIndex)
                    self.peis_freqs.append(freq)
            return (np.array([]), np.array([]), np.array([]))
        return self._get_converted_buffer()

    def _run(self):
        while True:
            self.latest_data = self._get_measurement_values()
            self._write_latest_data_to_file()
            if self.config.print_values:
                self._print_current_values()
            if self.current_values.State == 0:
                self.running = False
                print(f"CH{self.num}: Measure completed.")
                break
            self._monitoring_sequence_progression()
            if self._check_software_limits():
                print("Software limits met")
                self.end_technique()
            sleep_time = 0.02 if self.data_info.TechniqueID in self.PEIS_GEIS_TECH_IDS else 1
            time.sleep(sleep_time)

    def peis_geis_points(self):
        """(freqs, z, step_index) of the PEIS/GEIS points received so far, or None. Safe to call
        from another thread while the channel's thread keeps appending: a row count is
        snapshotted once and all three lists are sliced to it."""
        if not self.peis_freqs:
            return None
        n = len(self.peis_freqs)  # freq is appended last, so z/step already have >= n entries
        return (
            np.array(self.peis_freqs[:n], dtype=float),
            np.array(self.peis_z[:n], dtype=complex),
            np.array(self.peis_step_index[:n], dtype=int),
        )


def write_peis_geis_csv(path, freqs, z, step_index):
    """Write PEIS/GEIS results as CSV: technique_index, freq_Hz, Zre_ohm, Zim_ohm, absZ_ohm, phase_deg."""
    np.savetxt(
        path,
        np.column_stack([step_index, freqs, z.real, z.imag, np.abs(z), np.degrees(np.angle(z))]),
        delimiter=",", header="technique_index,freq_Hz,Zre_ohm,Zim_ohm,absZ_ohm,phase_deg", comments="",
    )
