import unittest

from elma.multisinegen import MultisineGenerator, MultisineGeneratorCombined


class FakeDevice:
    channel_num = 1

    def __init__(self):
        self.on = None
        self.calls = []

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False

    def select_awf(self, name):
        self.calls.append(("waveform", name))

    def set_amplitude(self, a):
        self.calls.append(("amplitude", a))

    def set_sample_rate(self, r):
        self.calls.append(("rate", r))

    def combine_channels(self):
        self.calls.append(("combine",))


class ForSteps(unittest.TestCase):
    def test_output_follows_the_flagged_steps(self):
        device = FakeDevice()
        generator = MultisineGenerator.for_steps(device, [0, 2, 3], "multisine", 1000.0, 0.05)
        states = []
        for index in range(5):
            generator.update(index)
            states.append(device.on)
        self.assertEqual(states, [True, False, True, True, False])

    def test_parameters_are_applied_when_switching_on(self):
        device = FakeDevice()
        generator = MultisineGenerator.for_steps(device, [1, 4], "ms", 250.0, 0.2)
        generator.update(4)
        self.assertEqual(device.calls, [("waveform", "ms"), ("amplitude", 0.2), ("rate", 250.0)])

    def test_lists_have_one_entry_per_step(self):
        generator = MultisineGenerator.for_steps(FakeDevice(), [0, 2, 5], "ms", 1.0, 0.1)
        self.assertEqual((len(generator.sequence_indexes), len(generator.names), len(generator.sample_rates),
                          len(generator.amplitudes)), (3, 3, 3, 3))

    def test_no_steps_means_always_off(self):
        device = FakeDevice()
        generator = MultisineGenerator.for_steps(device, [], "ms", 1.0, 0.1)
        generator.update(0)
        self.assertFalse(device.on)

    def test_combined_generator_follows_the_same_steps(self):
        low, high = FakeDevice(), FakeDevice()
        combined = MultisineGeneratorCombined(
            channel1=MultisineGenerator.for_steps(low, [0, 2], "ms", 10.0, 0.1),
            channel2=MultisineGenerator.for_steps(high, [0, 2], "ms", 1e6, 0.1),
            waveforms_names=["low", "high"],
        )
        combined.update(1)
        self.assertEqual((low.on, high.on), (False, False))
        combined.update(2)
        self.assertEqual((low.on, high.on), (True, True))


if __name__ == "__main__":
    unittest.main()
