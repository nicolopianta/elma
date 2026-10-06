import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elma.runs import AWGStepFollower, PotentiostatOnlyRun, RawCaptureRun, has_finished
from elma.multisinegen import MultisineGenerator


class FakeAWG:
    def __init__(self):
        self.on = None
        self.updates = []

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False

    def update(self, index):
        self.updates.append(index)


class FakeChannel:
    def __init__(self):
        self.running = False
        self.current_tech_index = 0
        self.started = self.stopped = 0
        self.ended = 0
        self.data_info = SimpleNamespace(TechniqueIndex=0)
        self.current_values = SimpleNamespace(Ewe=1.0)

    def start(self):
        self.running = True
        self.started += 1

    def stop(self):
        self.running = False
        self.stopped += 1

    def end_technique(self):
        self.ended += 1


def _wait(condition, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if condition():
            return True
        time.sleep(0.01)
    return False


class Follower(unittest.TestCase):
    def test_awg_is_updated_on_every_technique_change(self):
        awg, channel = FakeAWG(), FakeChannel()
        follower = AWGStepFollower(awg, channel, poll_interval=0.01)
        follower.start()
        self.assertEqual(awg.updates, [0])
        channel.current_tech_index = 2
        self.assertTrue(_wait(lambda: awg.updates == [0, 2]))
        channel.current_tech_index = 3
        self.assertTrue(_wait(lambda: awg.updates == [0, 2, 3]))
        follower.stop()
        n = len(awg.updates)
        channel.current_tech_index = 5
        time.sleep(0.05)
        self.assertEqual(len(awg.updates), n)

    def test_with_a_real_generator_the_output_follows_the_flags(self):
        class Device(FakeAWG):
            channel_num = 1
            def select_awf(self, n): pass
            def set_amplitude(self, a): pass
            def set_sample_rate(self, r): pass
        device, channel = Device(), FakeChannel()
        generator = MultisineGenerator.for_steps(device, [0, 2], "ms", 1.0, 0.1)
        follower = AWGStepFollower(generator, channel, poll_interval=0.01)
        follower.start()
        self.assertTrue(device.on)
        channel.current_tech_index = 1
        self.assertTrue(_wait(lambda: device.on is False))
        channel.current_tech_index = 2
        self.assertTrue(_wait(lambda: device.on is True))
        follower.stop()


class PotentiostatOnly(unittest.TestCase):
    def test_start_stop_without_a_follower_turns_the_awg_on_and_off(self):
        channel, awg = FakeChannel(), FakeAWG()
        run = PotentiostatOnlyRun(channel, [awg])
        run.start()
        self.assertTrue(run.running and awg.on)
        run.stop()
        self.assertFalse(run.running or awg.on)
        self.assertEqual((channel.started, channel.stopped), (1, 1))

    def test_with_a_follower_the_awg_follows_the_sequence(self):
        channel, awg = FakeChannel(), FakeAWG()
        run = PotentiostatOnlyRun(channel, [awg], awg=awg)
        run._awg_follower.poll_interval = 0.01
        run.start()
        channel.current_tech_index = 1
        self.assertTrue(_wait(lambda: awg.updates == [0, 1]))
        run.stop()
        self.assertFalse(awg.on)

    def test_software_limit_ends_the_technique(self):
        from elma.utils import ConditionAverage
        channel = FakeChannel()
        run = PotentiostatOnlyRun(channel, [], conditions=[ConditionAverage(0, "Ewe", ">", 0.5, 1)])
        run.start()
        self.assertTrue(_wait(lambda: channel.ended >= 1))
        run.stop()


class RawCapture(unittest.TestCase):
    def test_capture_is_saved_when_the_scope_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            channel, awg = FakeChannel(), FakeAWG()
            pico = SimpleNamespace(
                autoStopOuter=False, started=None, stopped=False, disconnected=False,
                run_streaming_non_blocking=lambda autoStop: setattr(pico, "started", autoStop),
                get_all_signals=lambda: (np.arange(5.0), np.arange(5.0) * 2),
                stop=lambda: setattr(pico, "stopped", True) or setattr(pico, "autoStopOuter", True),
                disconnect=lambda: setattr(pico, "disconnected", True),
            )
            run = RawCaptureRun(channel, pico, [awg], Path(tmp) / "exp")
            run.start()
            self.assertTrue(pico.started)                      # autoStop=True: the scope ends by itself
            self.assertFalse(has_finished(run))
            pico.autoStopOuter = True
            self.assertTrue(_wait(lambda: has_finished(run)))
            np.testing.assert_array_equal(np.load(Path(tmp) / "exp" / "raw_current.npy"), np.arange(5.0) * 2)
            run.stop()
            self.assertTrue(pico.disconnected and not awg.on)


class Finished(unittest.TestCase):
    def test_has_finished_duck_typing(self):
        self.assertTrue(has_finished(SimpleNamespace(running=False)))
        self.assertFalse(has_finished(SimpleNamespace(running=True)))
        self.assertFalse(has_finished(SimpleNamespace(finished=False, running=False)))   # `finished` wins
        self.assertTrue(has_finished(SimpleNamespace(finished=True)))


if __name__ == "__main__":
    unittest.main()
