# from attrs import define
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Optional, Union
import numpy as np
from numpy.fft import ifft, ifftshift

from npbuffer import NumpyCircularBuffer

from deistools.processing import MultiFrequencyAnalysis, FermiDiracFilter
from deistools.processing import detrending

from pyeclab import Channel

from elma._compat import patch_npbuffer
from elma.utils import log_online_analysis_error, log_online_analysis_timing


@dataclass
class ConditionAverageScope:
    """
    Quantity should be "current" or "voltage". Operator instead can only be ">" 
    or "<".
    """
    technique_index : int
    quantity : str
    operator : str
    threshold : float

@dataclass
class BlockCalculator:
    input_size : int
    sampling_time : float
    high_z_calculator : MultiFrequencyAnalysis 
    lp_filter : FermiDiracFilter
    ds_factor : int
    buffer_size : int
    potentiostat : Channel
    conditions: list[ConditionAverageScope] = field(default_factory=list)
    impedance_index: int = field(default= 0)
    save_dir : Optional[Union[str, Path]] = None  # experiment folder; when set, results are saved after every block
    impedance : np.array = field(init=False)
    voltage_ds: NumpyCircularBuffer = field(init=False)
    current_ds: NumpyCircularBuffer = field(init=False)
    
    def __post_init__(self):
        patch_npbuffer()  # no-op unless the installed npbuffer still has its data-loss bug
        self.save_dir = None if self.save_dir is None else Path(self.save_dir)
        self.high_z_calculator.compute_freq_axis()
        self.voltage_ds = NumpyCircularBuffer(self.buffer_size, np.float32)
        self.current_ds = NumpyCircularBuffer(self.buffer_size, np.float32)
        self.reset_impedance_memory()

    def _technique_folder(self) -> Path:
        # current_loop / current_tech_index are the Channel's trackers for the technique that is
        # running now -- the same values DEISchannel._execute_on_technique_termination uses for
        # its own folder name, so the end-of-technique save lands in this very folder.
        return self.save_dir / "pico_aquisition" / (
            f"cycle_{self.potentiostat.current_loop}_sequence_{self.potentiostat.current_tech_index}"
        )

    def calculate(self, data_voltage, data_current):
        """
        Process one window of voltage/current. With `save_dir` set it also (a) logs timing to
        logs/online_analysis_timing.log, (b) logs a failure, with traceback, to
        logs/online_analysis_errors.log before re-raising (the caller is a bare Thread that would
        swallow it), and (c) writes the decimated voltage/current and the impedance accumulated
        so far to <save_dir>/pico_aquisition/cycle_<loop>_sequence_<technique>/
        {voltage,current,impedance}.npy -- the same files and names deistools/elma write when a
        technique ends, which then simply overwrite them with identical content. Without it a
        crash or a manual Stop loses everything since the last technique end.
        """
        if self.save_dir is None:
            self._calculate_block(data_voltage, data_current)
            return
        log_online_analysis_timing(self.save_dir, f"calculate() started (impedance_index={self.impedance_index})")
        start = monotonic()
        try:
            self._calculate_block(data_voltage, data_current)
        except Exception:
            log_online_analysis_error(self.save_dir)
            log_online_analysis_timing(self.save_dir, "calculate() RAISED -- see online_analysis_errors.log")
            raise
        log_online_analysis_timing(
            self.save_dir,
            f"calculate() finished (impedance_index={self.impedance_index}, took {monotonic() - start:.2f}s)",
        )
        folder = self._technique_folder()
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "voltage.npy", self.voltage_ds.get_data())
        np.save(folder / "current.npy", self.current_ds.get_data())
        np.save(folder / "impedance.npy", self.impedance[:, :self.impedance_index])

    def _calculate_block(self, data_voltage, data_current):
        self.high_z_calculator.voltage, coordinates_voltage = detrending.remove_baseline(data_voltage, self.sampling_time)
        self.high_z_calculator.current, coordinates_current = detrending.remove_baseline(data_current, self.sampling_time)
        # self.high_z_calculator.voltage = data_voltage
        # self.high_z_calculator.current = data_current

        # Compute impedance of the high frequency band
        self.high_z_calculator.compute_fft()
        if self.high_z_calculator.freq_indexes == None:
            self.high_z_calculator.search_freq_indexes(
                self.high_z_calculator.ft_current
            )
        high_z = self.high_z_calculator.run_fft_eis()
        self.impedance[:,self.impedance_index] = high_z
        self.impedance_index += 1
        # Decimate the signals
        self.voltage_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_voltage * self.lp_filter.values)).real
        self.current_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_current * self.lp_filter.values)).real
        self.voltage_filt = detrending.redo_baseline(self.voltage_filt, coordinates_voltage)
        self.current_filt = detrending.redo_baseline(self.current_filt, coordinates_current)
        self.voltage_ds.push(self.voltage_filt[::self.ds_factor])
        self.current_ds.push(self.current_filt[::self.ds_factor])
        if self.check_software_limits_average_lin(coordinates_voltage, coordinates_current): # I have written below two possible methods to use here
            print("Software limit scope met!")
            self.potentiostat.end_technique()

    def save_results(self, directory):
        np.save(directory+'/impedance.npy', self.impedance[:,0:self.impedance_index])
        self.reset_impedance_memory()
        np.save(directory+'/voltage.npy', self.voltage_ds.empty())
        np.save(directory+'/current.npy', self.current_ds.empty())
        print('Data saved for the last technique!')

    def reset_impedance_memory(self):
        self.impedance = np.zeros(
            (self.high_z_calculator.frequencies.size, self.buffer_size), # This is too much allocation! 
            dtype = np.complex64,
            ) 
        self.impedance_index  = 0
    

    def check_software_limits_average_lin(self, conditions_voltage, conditions_current):
        """
        Check if a certain averege condition (< or > of a treshold value) is met for a
        value of the sampled data during the window of the online computation. It 
        uses the initial and final coordinate of the linearization. It adds the 
        value of the zero-frequency of the FFT to correct the linearized avarage.
        """
        mfa = self.high_z_calculator
        for condition in self.conditions:
            if self.potentiostat.data_info.TechniqueIndex == condition.technique_index:
                if condition.quantity == 'voltage':
                    value_avarage = (conditions_voltage['yfinish']+conditions_voltage['ystart'])/2
                    value_avarage = value_avarage + abs(mfa.ft_voltage[mfa.index_f0])
                    print(f'Avarage voltage is {value_avarage} V')
                if condition.quantity == 'current':
                    value_avarage = (conditions_current['yfinish']+conditions_current['ystart'])/2
                    value_avarage = value_avarage + abs(mfa.ft_current[mfa.index_f0])
                    print(f'Avarage current is {value_avarage} A')
                if condition.operator == ">" and value_avarage >= condition.threshold:
                    print(f'{condition.quantity} > {condition.threshold}')
                    return True
                elif condition.operator == "<" and value_avarage <= condition.threshold:
                    print(f'{condition.quantity} < {condition.threshold}')
                    return True
        return False


    def check_software_limits_average_fft(self):
        """
        Check if a certain averege condition (< or > of a treshold value) is met for a
        value of the sampled data during the window of the online computation. It 
        uses the zero-frequency value of the FFT as avarage.
        """
        mfa = self.high_z_calculator
        for condition in self.conditions:
            if self.potentiostat.data_info.TechniqueIndex == condition.technique_index:
                if condition.quantity == 'voltage':
                    value_avarage = abs(mfa.ft_voltage[mfa.index_f0])
                    print(f'Avarage voltage is {value_avarage} V')
                if condition.quantity == 'current':
                    value_avarage = abs(mfa.ft_current[mfa.index_f0])
                    print(f'Avarage current is {value_avarage} A')
                if condition.operator == ">" and value_avarage >= condition.threshold:
                    print(f'{condition.quantity} > {condition.threshold}')
                    return True
                elif condition.operator == "<" and value_avarage <= condition.threshold:
                    print(f'{condition.quantity} < {condition.threshold}')
                    return True
        return False