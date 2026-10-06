import unittest
from types import SimpleNamespace

import numpy as np

from elma.utils import ConditionAverage, check_software_limits


def _channel(ewe, technique_index=0):
    return SimpleNamespace(
        potentiostat=SimpleNamespace(
            data_info=SimpleNamespace(TechniqueIndex=technique_index),
            current_values=SimpleNamespace(Ewe=ewe),
        ),
    )


class ConditionAverageBuffer(unittest.TestCase):
    def test_buffer_keeps_full_precision(self):
        condition = ConditionAverage(0, "Ewe", ">", 2.6, 1)
        condition.buffer.push(np.array(2.6004))
        self.assertAlmostEqual(float(condition.buffer.get_data()[0]), 2.6004, places=9)

    def test_limit_triggers_for_a_millivolt_above_the_threshold(self):
        # with float16 (resolution ~2 mV at 2.6 V) 2.6004 V was stored as 2.6016 and
        # 2.5994 V as 2.5996: the comparison was off by up to a couple of mV
        channel = _channel(2.6004)
        channel.conditions = [ConditionAverage(0, "Ewe", ">", 2.6003, 1)]
        self.assertTrue(check_software_limits(channel))
        channel = _channel(2.6002)
        channel.conditions = [ConditionAverage(0, "Ewe", ">", 2.6003, 1)]
        self.assertFalse(check_software_limits(channel))

    def test_only_the_matching_technique_is_checked(self):
        channel = _channel(5.0, technique_index=1)
        channel.conditions = [ConditionAverage(0, "Ewe", ">", 1.0, 1)]
        self.assertFalse(check_software_limits(channel))


if __name__ == "__main__":
    unittest.main()
