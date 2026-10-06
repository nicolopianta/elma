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
    """
    pico : Union[Picoscope4000, Picoscope5000a]
    block_calculator : BlockCalculator
    potentiostat : Channel
    running : bool = field(default=False)
    deis_indexes : Optional[list] = None
    computation_thread : Thread = field(init=False)


    def __post_init__ (self):
        self.run_thread = Thread(target=self._run)
        self.block_calculator.running = True


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
