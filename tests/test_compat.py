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
