import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elma.deischannel import DEISchannel
from elma.multisinegen import MultisineGenerator
from elma.picocalculator import PicoCalculator


class FakePotentiostat:
    def __init__(self, folder):
        self.current_loop = 0
        self.current_tech_index = 0
        self.new_tech_index = 0
        self.function = None
        self.writer = SimpleNamespace(file_dir=Path(folder), experiment_name="exp")
        (Path(folder) / "exp").mkdir()


class FakePico:
    """A real PicoCalculator is a dataclass; a stand-in subclass keeps its isinstance() identity."""


class FakeAWGDevice:
    channel_num = 1

    def __init__(self):
        self.on = None

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False

    def select_awf(self, name):
        pass

    def set_amplitude(self, a):
        pass

    def set_sample_rate(self, r):
        pass


def _pico():
    block = SimpleNamespace(
        input_size=100, save_dir=None,
        high_z_calculator=SimpleNamespace(frequencies=np.array([1.0, 3.0])),
        lp_filter=SimpleNamespace(order=8, cutoff=10.0),
        saved=[],
    )
    block.save_results = lambda directory: block.saved.append(directory)
    pico = PicoCalculator(pico=SimpleNamespace(saving_dir="X", emptied=0), block_calculator=block,
                          potentiostat=SimpleNamespace())
    pico.pico.empty_buffers = lambda: setattr(pico.pico, "emptied", pico.pico.emptied + 1)
    pico.saved_subfolders = []
    pico.save_block_calculation = lambda sub: pico.saved_subfolders.append(sub)
    return pico


class StepFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pot = FakePotentiostat(self.tmp.name)
        self.pico = _pico()
        self.channel = DEISchannel(self.pot, self.pico, np.array([1.0, 3.0]), deis_indexes=[0, 2])

    def tearDown(self):
        self.tmp.cleanup()

    def test_flag_is_shared_with_the_pico_calculator(self):
        self.assertEqual(self.pico.deis_indexes, [0, 2])

    def test_only_flagged_steps_are_saved_and_the_others_drain_the_scope(self):
        for index in (0, 1, 2):
            self.pot.current_tech_index = index
            self.pot.function()                       # what Channel calls when a technique ends
        self.assertEqual(self.pico.saved_subfolders, ["/cycle_0_sequence_0", "/cycle_0_sequence_2"])
        self.assertEqual(self.pico.pico.emptied, 1)

    def test_no_flag_list_saves_everything_as_before(self):
        pico = _pico()
        channel = DEISchannel(self.pot, pico, np.array([1.0, 3.0]))
        self.assertIsNone(pico.deis_indexes)
        for index in (0, 1):
            self.pot.current_tech_index = index
            self.pot.function()
        self.assertEqual(pico.saved_subfolders, ["/cycle_0_sequence_0", "/cycle_0_sequence_1"])

    def test_metadata_records_the_flags_and_the_pico_settings(self):
        awg = MultisineGenerator.for_steps(FakeAWGDevice(), [0, 2], "multisine", 100.0, 0.05)
        self.channel.awg = awg
        self.channel.save_metadata()
        meta = json.loads((Path(self.tmp.name) / "exp" / "metadata_deis_exp.json").read_text())
        self.assertEqual(meta["DEIS steps (technique indexes)"], [0, 2])
        self.assertEqual(meta["Sequence indexes"], [0, 2])
        self.assertEqual(meta["STFFT-EIS frequencies (Hz)"], [1.0, 3.0])   # needs isinstance(), not type()==
        self.assertEqual(meta["Window length (Sa)"], 100)


if __name__ == "__main__":
    unittest.main()
