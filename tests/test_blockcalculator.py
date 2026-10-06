import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deistools.processing import FermiDiracFilter, MultiFrequencyAnalysis
from elma.blockcalculator import BlockCalculator


def _calculator(save_dir, potentiostat):
    fs, window = 1000.0, 2000                       # 2 s windows -> 0.5 Hz bins
    freqs = np.array([1.0, 3.0, 7.0])
    mfa = MultiFrequencyAnalysis(freqs, np.zeros(window), np.zeros(window), 1 / fs)
    mfa.compute_freq_axis()
    lp = FermiDiracFilter(mfa.freq_axis, 0, 20.0, 8)
    return BlockCalculator(input_size=window, sampling_time=1 / fs, high_z_calculator=mfa, lp_filter=lp,
                           ds_factor=10, buffer_size=5000, potentiostat=potentiostat, save_dir=save_dir), freqs


def _block(freqs, window=2000, fs=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(window) / fs
    v = sum(np.sin(2 * np.pi * f * t) for f in freqs) * 0.01 + rng.normal(0, 1e-4, window)
    i = sum(np.sin(2 * np.pi * f * t + 0.3) for f in freqs) * 1e-5 + rng.normal(0, 1e-7, window)
    return v, i


def _files(root):
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in Path(root).rglob("*") if p.is_file())


class IncrementalSaving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pot = SimpleNamespace(current_loop=0, current_tech_index=0, end_technique=lambda: None,
                                   data_info=SimpleNamespace(TechniqueIndex=0))

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_technique_gets_its_own_folder_and_nothing_is_duplicated(self):
        calc, freqs = _calculator(self.root, self.pot)
        for k in range(2):
            calc.calculate(*_block(freqs, seed=k))
        # technique 0 ends: deistools/elma's own save (also resets the impedance memory, drains the buffers)
        before = _files(self.root)
        calc.save_results(str(self.root / "pico_aquisition" / "cycle_0_sequence_0"))
        self.assertEqual(before, _files(self.root))        # it overwrote the incremental files
        self.pot.current_tech_index = 1
        calc.calculate(*_block(freqs, seed=2))
        v0 = np.load(self.root / "pico_aquisition/cycle_0_sequence_0/voltage.npy")
        z0 = np.load(self.root / "pico_aquisition/cycle_0_sequence_0/impedance.npy")
        v1 = np.load(self.root / "pico_aquisition/cycle_0_sequence_1/voltage.npy")
        z1 = np.load(self.root / "pico_aquisition/cycle_0_sequence_1/impedance.npy")
        self.assertEqual((v0.size, z0.shape), (2 * 200, (3, 2)))
        self.assertEqual((v1.size, z1.shape), (200, (3, 1)))
        self.assertEqual(sorted(p.name for p in (self.root / "pico_aquisition").iterdir()),
                         ["cycle_0_sequence_0", "cycle_0_sequence_1"])
        self.assertTrue((self.root / "logs" / "online_analysis_timing.log").exists())
        self.assertFalse((self.root / "impedance.npy").exists())

    def test_without_save_dir_nothing_is_written(self):
        calc, freqs = _calculator(None, self.pot)
        calc.calculate(*_block(freqs))
        self.assertEqual(calc.impedance_index, 1)
        self.assertEqual(_files(self.root), [])

    def test_a_failing_block_is_logged_and_re_raised(self):
        calc, freqs = _calculator(self.root, self.pot)
        with self.assertRaises(Exception):
            calc.calculate(np.zeros(10), np.zeros(10))       # wrong window size
        log = (self.root / "logs" / "online_analysis_errors.log").read_text()
        self.assertIn("Traceback", log)
        self.assertIn("RAISED", (self.root / "logs" / "online_analysis_timing.log").read_text())


if __name__ == "__main__":
    unittest.main()
