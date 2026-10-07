"""The installed dependencies must be the versions elma needs (the gui-support branches of the nicolopianta forks,
until the changes are merged upstream) -- these tests fail if one of them is the original from federicoscarpioni."""
import unittest

import numpy as np


class Npbuffer(unittest.TestCase):
    """push() must keep the last maxlen samples whatever the push sizes and after empty()."""

    @staticmethod
    def buffer(maxlen):
        from npbuffer import NumpyCircularBuffer
        return NumpyCircularBuffer(maxlen, np.float32)

    def test_pushes_that_end_flush_with_the_end_of_the_array(self):
        b = self.buffer(6)
        for k in range(4):
            b.push(np.arange(3 * k, 3 * k + 3, dtype=np.float32))
        self.assertEqual(b.get_data().tolist(), [6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

    def test_the_first_overflow_by_a_wrapping_push(self):
        b = self.buffer(6)
        b.push(np.arange(0, 4, dtype=np.float32))
        b.push(np.arange(4, 9, dtype=np.float32))
        self.assertEqual(b.get_data().tolist(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_a_run_after_empty_that_wraps_around_the_array_starts_clean(self):
        b = self.buffer(10)
        b.push(np.arange(0, 6, dtype=np.float32))
        b.empty()
        b.push(np.arange(10, 16, dtype=np.float32))
        self.assertEqual(b.get_data().tolist(), [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


class Pyeclab(unittest.TestCase):
    def test_peis_and_geis_come_with_pyeclab(self):
        import pyeclab
        from pyeclab.techniques import GEISTechnique, PEISTechnique
        self.assertTrue(hasattr(pyeclab, "PEISAwareChannel"))
        self.assertTrue(hasattr(pyeclab, "write_peis_geis_csv"))
        self.assertEqual((PEISTechnique.__name__, GEISTechnique.__name__), ("PEISTechnique", "GEISTechnique"))

    def test_the_default_hardware_uses_them(self):
        from elma import builder
        hardware = builder.default_hardware()
        import pyeclab
        self.assertIs(hardware.Channel, pyeclab.PEISAwareChannel)
        self.assertEqual(hardware.PEISTechnique.__module__, "pyeclab.techniques.peis")


class Multisine(unittest.TestCase):
    def test_imports_without_extra_packages(self):
        import multisine
        self.assertTrue(hasattr(multisine, "Multisine"))


if __name__ == "__main__":
    unittest.main()
