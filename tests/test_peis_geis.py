import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elma.peis_geis import GEISTechnique, PEISAwareChannel, PEISTechnique, write_peis_geis_csv


class FakeDevice:
    is_VMP3 = True


class Techniques(unittest.TestCase):
    def test_ecc_file_choice(self):
        peis = PEISTechnique.__new__(PEISTechnique)
        peis.device = FakeDevice()
        self.assertEqual(peis.choose_ecc_file(), "peis.ecc")
        peis.device = SimpleNamespace(is_VMP3=False)
        self.assertEqual(peis.choose_ecc_file(), "peis4.ecc")
        geis = GEISTechnique.__new__(GEISTechnique)
        geis.device = SimpleNamespace(is_VMP3=False)
        self.assertEqual(geis.choose_ecc_file(), "geis4.ecc")


class FakeBio:
    @staticmethod
    def ConvertNumericIntoSingle(value):
        return float(value)


def _channel():
    channel = PEISAwareChannel(FakeBio(), 1, writer=None, config=SimpleNamespace(print_values=False))
    return channel


class Decoding(unittest.TestCase):
    def test_process_1_rows_become_impedance_points_and_no_time_series_rows(self):
        channel = _channel()
        channel._get_data = lambda: None
        channel.data_info = SimpleNamespace(TechniqueID=104, ProcessIndex=1, NbRows=2, NbCols=4, TechniqueIndex=3)
        # rows: freq, |Ewe|, |I|, phase (rad)
        channel.data_buffer = [1000.0, 0.01, 0.001, -0.5, 100.0, 0.02, 0.001, -0.25]
        out = channel._get_measurement_values()
        self.assertEqual([a.size for a in out], [0, 0, 0])
        freqs, z, step = channel.peis_geis_points()
        np.testing.assert_allclose(freqs, [1000.0, 100.0])
        np.testing.assert_allclose(np.abs(z), [10.0, 20.0])
        np.testing.assert_allclose(np.angle(z), [-0.5, -0.25])
        np.testing.assert_array_equal(step, [3, 3])

    def test_process_0_of_a_peis_step_is_dropped(self):
        channel = _channel()
        channel._get_data = lambda: None
        channel.data_info = SimpleNamespace(TechniqueID=104, ProcessIndex=0, NbRows=5, NbCols=4, TechniqueIndex=0)
        channel.data_buffer = [0.0] * 20
        out = channel._get_measurement_values()
        self.assertEqual([a.size for a in out], [0, 0, 0])
        self.assertIsNone(channel.peis_geis_points())

    def test_other_techniques_use_the_normal_path(self):
        channel = _channel()
        channel._get_data = lambda: None
        channel.data_info = SimpleNamespace(TechniqueID=101, ProcessIndex=0, NbRows=1, NbCols=4, TechniqueIndex=0)
        channel._get_converted_buffer = lambda: ("normal",)
        self.assertEqual(channel._get_measurement_values(), ("normal",))

    def test_zero_current_gives_nan_not_an_exception(self):
        channel = _channel()
        channel._get_data = lambda: None
        channel.data_info = SimpleNamespace(TechniqueID=107, ProcessIndex=1, NbRows=1, NbCols=4, TechniqueIndex=1)
        channel.data_buffer = [10.0, 0.5, 0.0, 0.1]
        channel._get_measurement_values()
        self.assertTrue(np.isnan(channel.peis_z[0]))


class Csv(unittest.TestCase):
    def test_csv_columns_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.csv"
            freqs = np.array([1e5, 10.0])
            z = np.array([5 + 1j, 100 - 30j])
            write_peis_geis_csv(path, freqs, z, np.array([0, 1]))
            self.assertEqual(path.read_text().splitlines()[0], "technique_index,freq_Hz,Zre_ohm,Zim_ohm,absZ_ohm,phase_deg")
            table = np.loadtxt(path, delimiter=",", skiprows=1)
            np.testing.assert_array_equal(table[:, 2] + 1j * table[:, 3], z)
            np.testing.assert_array_equal(table[:, 0], [0, 1])


if __name__ == "__main__":
    unittest.main()
