"""
Display-only downsampling for very large plot arrays (e.g. a multisine
waveform at 7 decades: up to ~100M samples). Matplotlib rasterizing that
many points dominates "Generate multisine"'s total time once frequency
generation and waveform construction are both fast (measured: ~30s of a
~43s total at nd=7) -- this is a rendering cost, not a computation one,
so it's fixed on the display side, not by changing the underlying data.

Plain stride decimation (every Nth sample) is not used here because it can
silently miss the very features a plot exists to show -- e.g. a
multisine's crest-factor peak could fall on a skipped sample, making the
displayed waveform look lower-amplitude than it actually is. Min-max
bucket decimation avoids that: each bucket keeps both its minimum and
maximum sample (in original order, so the line doesn't zig-zag backward
in time/frequency), which preserves the visual envelope -- peaks and
troughs always survive -- while cutting the point count by orders of
magnitude.
"""
import numpy as np


def decimate_min_max(x, y, max_points=20000):
    """
    Downsample (x, y) to at most ~max_points points for display, keeping
    each bucket's min and max sample (envelope-preserving). No-op if y
    already has <= max_points samples. Vectorized (reshape + argmin/argmax
    along an axis, no Python-level loop over samples) -- sub-second even
    at 100M points.

    Returns
    -------
    (x_decimated, y_decimated) : np.ndarray, np.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = y.size
    if n <= max_points:
        return x, y

    n_buckets = max(1, max_points // 2)
    bucket_size = n // n_buckets
    usable = n_buckets * bucket_size

    y_buckets = y[:usable].reshape(n_buckets, bucket_size)
    x_buckets = x[:usable].reshape(n_buckets, bucket_size)
    min_idx = np.argmin(y_buckets, axis=1)
    max_idx = np.argmax(y_buckets, axis=1)
    rows = np.arange(n_buckets)

    min_x, max_x = x_buckets[rows, min_idx], x_buckets[rows, max_idx]
    min_y, max_y = y_buckets[rows, min_idx], y_buckets[rows, max_idx]
    # Emit min/max in whichever order they actually occur within the
    # bucket, so the line plot doesn't jump backward in x.
    first_is_min = min_idx <= max_idx

    out_x = np.empty(2 * n_buckets, dtype=x.dtype)
    out_y = np.empty(2 * n_buckets, dtype=y.dtype)
    out_x[0::2] = np.where(first_is_min, min_x, max_x)
    out_x[1::2] = np.where(first_is_min, max_x, min_x)
    out_y[0::2] = np.where(first_is_min, min_y, max_y)
    out_y[1::2] = np.where(first_is_min, max_y, min_y)

    if usable < n:
        # Leftover tail (< bucket_size samples) kept as-is, uncommon and
        # too small to matter for render cost.
        out_x = np.concatenate([out_x, x[usable:]])
        out_y = np.concatenate([out_y, y[usable:]])

    return out_x, out_y
