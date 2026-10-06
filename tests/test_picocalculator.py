import unittest
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np

from elma.picocalculator import PicoCalculator


class FakeBuffer:
    def __init__(self, n):
        self.n = n

    def get_length(self):
        return self.n

    def pop(self, size):
        self.n -= size
        return np.zeros(size)


class FakePico:
    def __init__(self, n=100):
        self.channels = {k: SimpleNamespace(buffer_total=FakeBuffer(n), vrange=1, conv_factor=1) for k in "AB"}
        self.emptied = 0

    def convert_ADC_numbers(self, data, vrange, conv):
        return data

    def empty_buffers(self):
        self.emptied += 1
        for ch in self.channels.values():
            ch.buffer_total.n = 0


class FakeBlockCalculator:
    input_size = 10
    save_dir = None

    def __init__(self):
        self.calls = 0
        self.release = Event()
        self.release.set()

    def calculate(self, v, i):
        self.release.wait(2)
        self.calls += 1


def _calc(tech_index=0, tech_id=101, deis_indexes=None, n=100):
    pot = SimpleNamespace(current_tech_index=tech_index, current_tech_id=tech_id)
    block = FakeBlockCalculator()
    return PicoCalculator(pico=FakePico(n), block_calculator=block, potentiostat=pot, deis_indexes=deis_indexes), pot, block


class StepGating(unittest.TestCase):
    def test_default_rule_skips_ocv_only(self):
        calc, pot, block = _calc(tech_id=100)                 # OCV
        calc._poll_once()
        self.assertEqual(calc.pico.emptied, 1)
        calc, pot, block = _calc(tech_id=101)                 # CA
        calc._poll_once()
        calc.computation_thread.join()
        self.assertEqual(block.calls, 1)

    def test_deis_indexes_decide_instead_of_the_technique_type(self):
        calc, pot, block = _calc(tech_index=1, tech_id=101, deis_indexes=[0, 2])
        calc._poll_once()
        self.assertEqual(calc.pico.emptied, 1)                # a CA step that is not flagged: data discarded
        self.assertFalse(hasattr(calc, "computation_thread"))
        calc, pot, block = _calc(tech_index=2, tech_id=100, deis_indexes=[0, 2])
        calc._poll_once()                                     # an OCV step that IS flagged: processed
        calc.computation_thread.join()
        self.assertEqual(block.calls, 1)


class NoOverlap(unittest.TestCase):
    def test_no_new_calculation_while_the_previous_one_runs(self):
        calc, pot, block = _calc(n=100)
        block.release.clear()                                 # calculate() blocks
        popped = calc._poll_once()
        self.assertEqual(popped, 1)
        popped = calc._poll_once(popped)                      # data is waiting, but the thread is still busy
        self.assertEqual(popped, 1)
        self.assertEqual(calc.pico.channels["A"].buffer_total.n, 90)
        block.release.set()
        calc.computation_thread.join()
        popped = calc._poll_once(popped)
        calc.computation_thread.join()
        self.assertEqual((popped, block.calls), (2, 2))

    def test_not_enough_data_pops_nothing(self):
        calc, pot, block = _calc(n=10)                        # window is 10: needs strictly more
        self.assertEqual(calc._poll_once(), 0)


if __name__ == "__main__":
    unittest.main()
