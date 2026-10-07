import unittest

import numpy as np
from npbuffer import NumpyCircularBuffer

from elma import _compat


class NpbufferPatch(unittest.TestCase):
    def test_rolling_window_matches_a_plain_sliding_window(self):
        _compat.patch_npbuffer()
        rng = np.random.default_rng(0)
        for maxlen in (6, 15, 40):
            buffer = NumpyCircularBuffer(maxlen, np.float32)
            reference = []
            counter = 0.0
            for _ in range(200):
                size = int(rng.choice([1, 2, 3, 5, maxlen // 3 or 1, maxlen // 2 or 1]))
                data = np.arange(counter, counter + size, dtype=np.float32)
                counter += size
                buffer.push(data)
                reference = (reference + data.tolist())[-maxlen:]
                self.assertEqual(buffer.get_data().tolist(), reference)

    def test_the_flush_with_the_end_case_that_lost_data(self):
        _compat.patch_npbuffer()
        buffer = NumpyCircularBuffer(6, np.float32)
        for k in range(4):
            buffer.push(np.arange(3 * k, 3 * k + 3, dtype=np.float32))
        self.assertEqual(buffer.get_data().tolist(), [6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

    def test_the_first_overflow_by_a_wrapping_push_case(self):
        _compat.patch_npbuffer()
        buffer = NumpyCircularBuffer(6, np.float32)
        buffer.push(np.arange(0, 4, dtype=np.float32))
        buffer.push(np.arange(4, 9, dtype=np.float32))
        self.assertEqual(buffer.get_data().tolist(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_a_run_after_empty_that_wraps_around_the_array_starts_clean(self):
        # the end of every technique empties the buffer; the next technique then starts mid-array
        _compat.patch_npbuffer()
        buffer = NumpyCircularBuffer(10, np.float32)
        buffer.push(np.arange(0, 6, dtype=np.float32))
        self.assertEqual(buffer.empty().tolist(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        buffer.push(np.arange(10, 16, dtype=np.float32))          # wraps, 6 samples stored
        self.assertEqual(buffer.get_data().tolist(), [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        self.assertEqual(buffer.empty().tolist(), [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    def test_matches_a_plain_list_for_mixed_push_pop_and_empty(self):
        _compat.patch_npbuffer()
        rng = np.random.default_rng(1)
        for maxlen in (5, 8, 13):
            buffer, reference, n = NumpyCircularBuffer(maxlen, np.float32), [], 0
            for _ in range(400):
                action = int(rng.integers(0, 4))
                if action < 2:
                    size = int(rng.integers(0, maxlen + 1))
                    buffer.push(np.arange(n, n + size, dtype=np.float32))
                    reference = (reference + list(range(n, n + size)))[-maxlen:]
                    n += size
                elif action == 2 and reference:
                    count = int(rng.integers(1, len(reference) + 1))
                    self.assertEqual(buffer.pop(count).tolist(), reference[:count])
                    reference = reference[count:]
                else:
                    self.assertEqual(buffer.empty().tolist(), reference)
                    reference = []
                self.assertEqual(buffer.get_data().tolist(), reference)

    def test_block_calculator_applies_the_patch_itself(self):
        # user scripts never call patch_npbuffer: building a BlockCalculator is enough
        from deistools.processing import FermiDiracFilter, MultiFrequencyAnalysis
        from elma.blockcalculator import BlockCalculator
        _compat._npbuffer_checked = False
        mfa = MultiFrequencyAnalysis(np.array([1.0]), np.zeros(100), np.zeros(100), 0.01)
        calc = BlockCalculator(input_size=100, sampling_time=0.01, high_z_calculator=mfa,
                               lp_filter=FermiDiracFilter(np.zeros(100), 0, 1, 2), ds_factor=10,
                               buffer_size=6, potentiostat=None)
        calc.voltage_ds.push(np.arange(0, 4, dtype=np.float32))
        calc.voltage_ds.push(np.arange(4, 9, dtype=np.float32))
        self.assertEqual(calc.voltage_ds.get_data().tolist(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


if __name__ == "__main__":
    unittest.main()
