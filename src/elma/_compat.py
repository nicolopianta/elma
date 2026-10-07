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
    Idempotent; does nothing if the installed npbuffer is already correct. The same fix is in the
    npbuffer repository itself (branch gui-support of the fork); this patch goes away once it is released.

    Whether the oldest samples are overwritten depends on what is stored plus what comes in, not
    on whether the write crosses the end of the array: once the buffer is full ("overflown"),
    head and tail stay in lockstep (every push overwrites as much old data as it writes); before
    that the head stays where pop()/empty() left it. The head must NOT be moved just because a
    push wraps: after empty() (the end of every technique) the next run starts mid-array.
    1. The "wrap" branch (the push crosses the end of the array) only moved the head when the
       buffer was already overflown, so the push that overflows it for the first time left the
       head behind and get_data() returned only the samples up to the tail (push 4 then 5 into a
       buffer of 6 returned 2 samples). It matters whenever the buffer size is not a multiple of
       the push size.
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
        total = self.get_length() + new_data.size
        if self._tail + new_data.size > self.maxlen:
            split = self.maxlen - self._tail
            self._data[self._tail:self.maxlen] = new_data[:split]
            self._data[0:new_data.size - split] = new_data[split:]
        else:
            self._data[self._tail:self._tail + new_data.size] = new_data
        self._tail = (self._tail + new_data.size) % self.maxlen
        if total > self.maxlen:
            self._head = self._tail  # the oldest samples were overwritten
        self._overflown = total >= self.maxlen

    NumpyCircularBuffer.push = fixed_push
    return True
