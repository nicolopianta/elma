from collections import deque
from math import ceil
from pathlib import Path
import numpy as np
from threading import Thread
from time import sleep
from dataclasses import dataclass, field
from typing import Optional, Union

from pypicostreaming import Picoscope4000, Picoscope5000a

from elma.blockcalculator import BlockCalculator
from elma.utils import log_online_analysis_timing

from pyeclab import Channel


@dataclass
class PicoCalculator:
    """
    Streams the Picoscope and hands one window at a time to the block calculator.

    deis_indexes: technique indexes (positions in the potentiostat sequence) during which DEIS
    runs. Windows are only processed during those steps; during the others the scope buffers
    are emptied every cycle. None keeps the old rule: every technique except OCV.
    A new calculation is never started while the previous one is still running: they mutate
    shared, unsynchronised state in the block calculator (voltage, current, ft_voltage, ...),
    and two overlapping calls corrupted each other's intermediate data.

    skip_start_seconds / skip_end_seconds: the multisine does not start exactly with the step and the end of a step is
    noticed late, so the scope data within these many seconds of the start and of the end of a DEIS step are not
    trustworthy. The windows that overlap them by more than EDGE_TOLERANCE of their length are discarded: the first
    ones are popped and dropped, and the last ones are held back until a later window proves that the step was still
    running, so they are never calculated. Slow multisines (windows of minutes) lose nothing, because the overlap is
    a small fraction of the window. 0 (default) keeps every window.
    """
    EDGE_TOLERANCE = 0.02
    pico : Union[Picoscope4000, Picoscope5000a]
    block_calculator : BlockCalculator
    potentiostat : Channel
    running : bool = field(default=False)
    deis_indexes : Optional[list] = None
    skip_start_seconds : float = 0.0
    skip_end_seconds : float = 0.0
    computation_thread : Thread = field(init=False)


    def __post_init__ (self):
        self.run_thread = Thread(target=self._run)
        self.block_calculator.running = True
        self._edge_index = None   # technique index of the step whose windows are being popped
        self._skip_left = 0       # windows still to drop at the start of that step
        self._held = deque()      # the latest windows, held back in case the step has already ended

    def _windows_to_skip(self, seconds):
        """How many whole windows cover `seconds` of data, or 0 if that is a negligible part of one."""
        window = self.block_calculator.input_size * getattr(self.block_calculator, 'sampling_time', 0.0)
        if seconds <= 0 or window <= 0 or seconds / window <= self.EDGE_TOLERANCE:
            return 0
        return ceil(seconds / window - 1e-9)

    def _forget_step(self):
        self._edge_index = None
        self._held.clear()


    def start(self):
        self.pico.run_streaming_non_blocking(autoStop = False)
        self.running = True
        self.run_thread.start()


    def stop(self):
        self.pico.stop()
        self.pico.disconnect()
        self.running = False

    def empty_buffers(self):
        self.pico.empty_buffers()

    def save_block_calculation(self, subfolder_name):
        self._forget_step()    # the windows held back at the end of the step are never calculated
        if hasattr(self, 'computation_thread') and self.computation_thread.is_alive():
            self.computation_thread.join()
        saving_file_path = self.pico.saving_dir + subfolder_name
        Path(saving_file_path).mkdir(parents=True, exist_ok=True)
        self.block_calculator.save_results(saving_file_path)
        self.asked_saving = False


    def _step_is_active(self):
        if self.deis_indexes is None:
            return self.potentiostat.current_tech_id != 100  # not OCV
        return self.potentiostat.current_tech_index in self.deis_indexes


    def _poll_once(self, blocks_popped=0):
        """One polling cycle. Returns the updated count of popped blocks."""
        save_dir = getattr(self.block_calculator, 'save_dir', None)
        if not self._step_is_active():
            self._forget_step()
            self.empty_buffers()
            return blocks_popped
        if hasattr(self, 'computation_thread') and self.computation_thread.is_alive():
            log_online_analysis_timing(save_dir, 'poll: previous calculate() still running, skipping this cycle')
            return blocks_popped
        data_lengthA = self.pico.channels['A'].buffer_total.get_length()
        data_lengthB = self.pico.channels['B'].buffer_total.get_length()
        size = self.block_calculator.input_size
        if data_lengthA > size and data_lengthB > size:
            blocks_popped += 1
            log_online_analysis_timing(
                save_dir,
                f'block {blocks_popped}: popping window_size={size} samples '
                f'(bufferA had {data_lengthA}, bufferB had {data_lengthB})',
            )
            self.voltage_block = self.pico.convert_ADC_numbers(
                self.pico.channels['A'].buffer_total.pop(size),
                self.pico.channels['A'].vrange,
                self.pico.channels['A'].conv_factor
            )
            self.current_block = self.pico.convert_ADC_numbers(
                self.pico.channels['B'].buffer_total.pop(size),
                self.pico.channels['B'].vrange,
                self.pico.channels['B'].conv_factor
            )
            index = self.potentiostat.current_tech_index
            if index != self._edge_index:  # first window of a new step
                self._edge_index = index
                self._skip_left = self._windows_to_skip(self.skip_start_seconds)
                self._held.clear()
            if self._skip_left > 0:
                self._skip_left -= 1
                log_online_analysis_timing(save_dir, f'block {blocks_popped}: start of the step, dropped')
                return blocks_popped
            hold = self._windows_to_skip(self.skip_end_seconds)
            if hold:
                self._held.append((self.voltage_block, self.current_block))
                if len(self._held) <= hold:
                    return blocks_popped
                self.voltage_block, self.current_block = self._held.popleft()
            self.computation_thread = Thread(target=self.block_calculator.calculate, args=(self.voltage_block, self.current_block,))
            self.computation_thread.start()
        return blocks_popped


    def _run(self):
        save_dir = getattr(self.block_calculator, 'save_dir', None)
        log_online_analysis_timing(save_dir, 'poll loop started')
        blocks_popped = 0
        while self.running:
            blocks_popped = self._poll_once(blocks_popped)
            sleep(1)
        log_online_analysis_timing(
            save_dir, f'poll loop exited (self.running became False); {blocks_popped} block(s) popped in total'
        )
