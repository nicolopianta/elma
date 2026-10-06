import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from multisine import Multisine

from elma.design import (
    amplitudes_from_impedance,
    equal_amplitudes,
    generate_imd_safe_frequencies,
    load_multisine_json,
    load_split_multisine_bands,
    multisine_from_dict,
    rescale_multisine_frequency,
    save_multisine_json,
    save_split_multisine_json,
    split_multisine,
)

FREQS = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0])


def _design():
    return Multisine(1000.0, FREQS, np.ones(FREQS.size), phases=np.linspace(0, 1, FREQS.size))


class Persistence(unittest.TestCase):
    def test_single_design_round_trip(self):
        ms = _design()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "single.json"
            save_multisine_json(ms, path)
            data = load_multisine_json(path)
        self.assertEqual(data["type"], "single")
        back = multisine_from_dict(data)
        np.testing.assert_allclose(back.frequencies, ms.frequencies)
        np.testing.assert_allclose(back.phases, ms.phases)
        self.assertEqual(back.number_points, ms.number_points)

    def test_split_design_keeps_the_high_bands_own_number_of_points(self):
        low, high = split_multisine(_design(), 10.0)
        np.testing.assert_allclose(low.frequencies, [1, 2, 3, 5, 8])
        np.testing.assert_allclose(high.frequencies, [13, 21, 34])
        self.assertNotEqual(low.number_points, high.number_points)       # the pitfall the JSON field guards against
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            save_split_multisine_json(low, high, 10.0, path)
            data = load_multisine_json(path)
            self.assertEqual(data["type"], "split")
            low2, high2 = load_split_multisine_bands(data)
            self.assertEqual((low2.number_points, high2.number_points), (low.number_points, high.number_points))
            # an older file without number_points: the high band's is recomputed
            del data["high_band"]["number_points"]
            _, high3 = load_split_multisine_bands(data)
            self.assertEqual(high3.number_points, high.number_points)


class Scaling(unittest.TestCase):
    def test_rescale_remaps_the_reference_frequency(self):
        out = rescale_multisine_frequency(_design(), 0.01, 1.0)
        np.testing.assert_allclose(out.frequencies, FREQS * 0.01)
        self.assertAlmostEqual(out.sampling_frequency, 10.0)
        self.assertEqual(out.number_points, _design().number_points)
        with self.assertRaises(ValueError):
            rescale_multisine_frequency(_design(), 0.0, 1.0)


class Frequencies(unittest.TestCase):
    def test_imd_safe_frequencies_start_at_the_first_frequency(self):
        f = generate_imd_safe_frequencies(nd=2, ppd=4, order=3, first_freq=2.0)
        self.assertEqual(f[0], 2.0)
        self.assertTrue(np.all(np.diff(f) > 0))

    def test_amplitudes(self):
        np.testing.assert_array_equal(equal_amplitudes(FREQS), np.ones(FREQS.size))


class Imports(unittest.TestCase):
    def test_importing_the_package_does_not_need_lmfit(self):
        code = "import sys; import elma.design; print('lmfit' in sys.modules)"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False", out.stderr)


if __name__ == "__main__":
    unittest.main()
