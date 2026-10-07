import unittest
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np

from elma.picocalculator import PicoCalculator


class FakeBuffer:
    def __init__(self, n):
        self.n = n
        self.popped = 0

    def get_length(self):
        return self.n

    def pop(self, size):
        self.n -= size
        self.popped += 1
        return np.full(size, float(self.popped))     # the k-th window is filled with k


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
    sampling_time = 0.1      # a window is 1 s
    save_dir = None

    def __init__(self):
        self.calls = 0
        self.windows = []
        self.release = Event()
        self.release.set()

    def calculate(self, v, i):
        self.release.wait(2)
        self.calls += 1
        self.windows.append(float(v[0]))


def _calc(tech_index=0, tech_id=101, deis_indexes=None, n=100, **kw):
    pot = SimpleNamespace(current_tech_index=tech_index, current_tech_id=tech_id)
    block = FakeBlockCalculator()
    return PicoCalculator(pico=FakePico(n), block_calculator=block, potentiostat=pot, deis_indexes=deis_indexes, **kw), pot, block


def _run_polls(calc, count):
    popped = 0
    for _ in range(count):
        popped = calc._poll_once(popped)
        if hasattr(calc, "computation_thread"):
            calc.computation_thread.join()
    return popped


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


class StepEdges(unittest.TestCase):
    """Windows that overlap the start / end of a step by more than 2 % of their length are never calculated."""

    def test_default_keeps_every_window(self):
        calc, pot, block = _calc(n=1000)
        _run_polls(calc, 6)
        self.assertEqual(block.windows, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_the_first_and_the_last_windows_of_a_step_are_dropped(self):
        calc, pot, block = _calc(n=1000, skip_start_seconds=1.5, skip_end_seconds=1.5)   # windows of 1 s: 2 at each edge
        _run_polls(calc, 8)                                   # 8 windows popped
        self.assertEqual(block.windows, [3.0, 4.0, 5.0, 6.0])      # 1, 2 dropped at the start; 7, 8 still held at the end

    def test_the_held_windows_are_discarded_when_the_step_ends(self):
        calc, pot, block = _calc(n=1000, skip_start_seconds=0.5, skip_end_seconds=1.0)
        _run_polls(calc, 4)
        self.assertEqual(block.windows, [2.0, 3.0])           # window 1 dropped at the start, window 4 held back
        calc._forget_step()                                   # what save_block_calculation does first
        self.assertEqual(len(calc._held), 0)

    def test_every_new_step_drops_its_own_first_windows(self):
        calc, pot, block = _calc(n=1000, deis_indexes=[0, 1], skip_start_seconds=0.5)
        _run_polls(calc, 3)                                   # step 0: window 1 dropped, 2 and 3 calculated
        calc._forget_step()                                   # step 0 ended and was saved
        pot.current_tech_index = 1
        _run_polls(calc, 3)                                   # step 1: window 4 dropped, 5 and 6 calculated
        self.assertEqual(block.windows, [2.0, 3.0, 5.0, 6.0])

    def test_a_step_that_is_not_flagged_resets_the_edge_tracking(self):
        calc, pot, block = _calc(n=1000, deis_indexes=[0, 2], skip_start_seconds=0.5)
        _run_polls(calc, 2)                                   # step 0: window 1 dropped, 2 calculated
        pot.current_tech_index = 1                            # an unflagged step: data discarded
        calc._poll_once()
        pot.current_tech_index = 2
        calc.pico.channels["A"].buffer_total.n = 100
        calc.pico.channels["B"].buffer_total.n = 100
        _run_polls(calc, 2)                                   # step 2: its first window is dropped again
        self.assertEqual(block.windows, [2.0, 4.0])

    def test_slow_multisines_lose_nothing(self):
        calc, pot, block = _calc(n=100000, skip_start_seconds=1.5, skip_end_seconds=1.5)
        calc.block_calculator.input_size = 10
        calc.block_calculator.sampling_time = 10.0            # windows of 100 s: 1.5 s is 1.5 % of one
        self.assertEqual((calc._windows_to_skip(1.5), calc._windows_to_skip(3.0)), (0, 1))
        _run_polls(calc, 3)
        self.assertEqual(block.windows, [1.0, 2.0, 3.0])

    def test_window_counts(self):
        calc, pot, block = _calc()
        self.assertEqual([calc._windows_to_skip(s) for s in (0.0, 0.01, 0.5, 1.0, 1.5, 2.5)], [0, 0, 1, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
