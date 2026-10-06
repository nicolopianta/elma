"""
Workarounds for bugs in third-party packages that elma depends on, applied lazily and
only while the installed package still has the bug.
"""
import numpy as np

_npbuffer_checked = False


def _npbuffer_push_is_buggy(NumpyCircularBuffer) -> bool:
    """Runs the two scenarios that expose the bugs on a tiny buffer (see patch_npbuffer):
    four pushes of 3 samples into a buffer of 6 must leave the last 6 samples (the buggy
    version leaves 3), and a push of 4 then of 5 must leave the last 6 of 9 (the buggy
    version leaves 3)."""
    buffer = NumpyCircularBuffer(6, np.float32)
    for k in range(4):
        buffer.push(np.arange(3 * k, 3 * k + 3, dtype=np.float32))
    if buffer.get_data().tolist() != [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]:
        return True
    buffer = NumpyCircularBuffer(6, np.float32)
    buffer.push(np.arange(0, 4, dtype=np.float32))
    buffer.push(np.arange(4, 9, dtype=np.float32))
    return buffer.get_data().tolist() != [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def patch_npbuffer():
    """
    Fix two data-loss bugs in `npbuffer.NumpyCircularBuffer.push()` (used for the decimated
    voltage/current of BlockCalculator, where we have no hook to substitute another class).
    Idempotent; does nothing if the installed npbuffer is already correct.

    Once the buffer is full ("overflown"), head and tail must stay in lockstep -- every push
    overwrites exactly as much old data as it writes.
    1. The "wrap" branch (the push crosses the end of the array) only moved the head when the
       buffer was already overflown. A push that wraps is by definition the one that overflows
       it for the first time, so the head was left at 0 while the tail moved on, and
       get_data() returned only the samples up to the tail (push 4 then 5 into a buffer of 6
       returned 2 samples). It matters whenever the buffer size is not a multiple of the push
       size. The head now always follows the tail there.
    2. The "no-wrap" branch instead did
       `self._head = self._tail if self._tail > self._head else self._head`: when a push lands
    exactly flush with the end of the array, the new tail wraps to 0 (`% maxlen`), 0 is never
    ">" a non-zero old head, and the head is left behind. get_data() then returns only the
    last push instead of the whole rolling window. It fires whenever maxlen is a multiple of
    the push size (buffer_size = buffer_duration x resampling_frequency and one decimated
    block is window_size/ds_factor samples), e.g. a 1000 s run with 9 blocks of 5000 samples
    in a 15000-sample buffer lost 2 of the last 3 blocks just before the end-of-run save.
    """
    global _npbuffer_checked
    if _npbuffer_checked:
        return False
    _npbuffer_checked = True
    from npbuffer import NumpyCircularBuffer

    if not _npbuffer_push_is_buggy(NumpyCircularBuffer):
        return False

    def fixed_push(self, new_data):
        if new_data.size > self.maxlen:
            raise Exception("Data input is longer than buffer size. Operation not possible.")
        if self._tail + new_data.size > self.maxlen:
            overflow = (self._tail + new_data.size) - self.maxlen
            self._data[self._tail:self.maxlen] = new_data[:self.maxlen - self._tail]
            self._data[0:overflow] = new_data[self.maxlen - self._tail:]
            self._tail = overflow
            self._head = self._tail
            self._overflown = True
        else:
            self._data[self._tail:self._tail + new_data.size] = new_data
            self._tail = (self._tail + new_data.size) % self.maxlen
            if self._overflown:
                self._head = self._tail
        if self._tail == self._head:
            self._overflown = True

    NumpyCircularBuffer.push = fixed_push
    return True
