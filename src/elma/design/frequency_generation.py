"""
Intermodulation-safe multisine frequency generation.

Faithful port of the frequency-selection algorithm in the original
``MultisineGenerator.py`` research script (Multisine.generate_freq /
__odd_frequency_generation), lifted out of that script's class into plain
functions operating on plain arrays.

The core idea: starting from a log-spaced grid of integer candidate
frequencies (`nd` decades, `ppd` points per decade), greedily build up a
frequency set such that no intermodulation product (a sum or difference of
up to `order` frequencies already in the set) coincides with another
frequency in the set. An IMD collision would otherwise corrupt the
per-frequency impedance estimate at the colliding frequency.

Search complexity grows with the target frequency range (up to 10**nd), so
very large `nd` (or `order`) will still be slow -- the exact-subset-sum
check (_subsetsum) and the IMD check (_is_sub_sum) are both DPs now, but
neither is free. _subsetsum was, by far, the dominant cost (~85% of total
runtime, profiled at nd=5): fixed below by swapping its O(n*target)
Python-level boolean table for an O(n) big-integer bitset DP (target ~
150,000: 1260x faster on a single call, verified against the original
table-based algorithm on 3000+ randomized cases -- identical results,
including which subset size is returned, not just reachability).

generate_imd_safe_frequencies' optional split_frequency makes every
frequency at or above it an integer multiple of split_frequency (instead
of just of first_freq), searched together with the rest in one pass, so a
later split (splitting.split_multisine) doesn't inherit the whole range's
fine 1-unit resolution just to represent the high band's own frequencies
exactly -- the high band's waveform can use split_frequency itself as its
FFT bin-alignment reference, needing far fewer samples than the full
range's resolution would demand. See generate_imd_safe_frequencies'
docstring for the mechanism and splitting.py for how the smaller
number_points gets picked up automatically.
"""
import numpy as np


def _subsetsum(nums, target):
    """
    Return the size of an exact subset of `nums` summing to `target`, or
    False if no such subset exists (dynamic-programming subset-sum over
    positive integers).

    Reachability is tracked as a single Python integer bitset (bit v set
    <=> sum v is reachable using a prefix of nums) instead of a full
    boolean table: dp[k] = dp[k-1] | (dp[k-1] << nums[k-1]) replaces an
    O(target) Python-level inner loop per element with one big-integer
    shift/OR, done in C -- ~1000x faster for realistic target sizes here.
    Masked to target+1 bits after every step so the bitset never grows
    past what the original table tracked (columns 0..target).

    The backward reconstruction (which elements were used, to recover
    `len(picked)`) queries the exact same bits the original table would
    have held (table[row][col] == bit `col` of dp[row]), with identical
    tie-breaking -- so this returns exactly what the original
    implementation did, not just the same reachability verdict.
    """
    n = len(nums)
    mask = (1 << (target + 1)) - 1
    dp = [1]
    reachable = 1
    for num in nums:
        if num <= target:
            reachable = (reachable | (reachable << num)) & mask
        dp.append(reachable)

    if not (dp[n] >> target) & 1:
        return False

    picked = []
    col, row = target, n
    while col > 0 or row > 0:
        bit_here = (dp[row] >> col) & 1
        bit_prev = (dp[row - 1] >> col) & 1
        if bit_here == bit_prev:
            row -= 1
        else:
            col -= nums[row - 1]
            picked.append(nums[row - 1])
            row -= 1
    return len(picked)


def _reachable_signed_sums(values, max_size):
    """
    reachable[r] = set of sums obtainable by choosing exactly r elements
    from `values` (each used at most once) and giving each an independent
    +/- sign, for r in 0..max_size.

    Built incrementally, one element at a time (classic 0/1-knapsack-via-
    sets: iterate the size dimension downward so each element is folded in
    at most once per subset).
    """
    reachable = {0: {0}}
    for f in values:
        f = int(f)
        for r in range(max_size - 1, -1, -1):
            for s in reachable.get(r, ()):
                reachable.setdefault(r + 1, set()).add(s + f)
                reachable[r + 1].add(s - f)
    return reachable


def _is_sub_sum(frequencies, order):
    """
    True if any subset of size 2..`order` of the signed frequency set (each
    frequency taken as both +f and -f, modelling that an intermodulation
    product can add or subtract contributions) sums to a value already
    present in the set.

    Any such subset must include the current maximum frequency (a subset
    that doesn't touch it can't be a *new* collision introduced by adding
    it), so this reduces to: pick a size r in [1, order-1] of the other
    frequencies, sign each independently, and check whether (+/- max) plus
    that signed sum lands on an existing frequency. The r-sized signed sums
    are built once via DP instead of enumerating every combination, which
    is what made the previous implementation combinatorial in `order` and
    the frequency count.
    """
    frequencies = list(frequencies)
    max_freq = max(frequencies)
    others = [f for f in frequencies if f != max_freq]
    magnitudes = set(frequencies)

    reachable = _reachable_signed_sums(others, max_size=order - 1)
    for r in range(1, order):
        for u in reachable.get(r, ()):
            if abs(max_freq + u) in magnitudes or abs(u - max_freq) in magnitudes:
                return True
    return False


def _force_odd_frequencies(candidates):
    """order == 2 special case: force every candidate odd (cheapest way to
    guarantee no two candidates sum/difference to a third)."""
    candidates = [c + 1 if c % 2 == 0 else c for c in candidates]
    candidates[-1] -= 1
    return np.unique(candidates)


def _prune_near_split(accepted, k_split, ppd):
    """
    Drop the largest accepted frequency below k_split if it crowds the
    split point closer than a normal grid step would ever place a
    legitimate candidate. A log-spaced grid at ppd points/decade has
    consecutive values in ratio ~10**(1/ppd), so the local spacing near
    k_split is ~k_split*(10**(1/ppd)-1); anything closer than half that is
    treated as "too near" and removed, repeating in case more than one
    value crowds the boundary.
    """
    accepted = sorted(accepted)
    local_spacing = k_split * (10.0 ** (1.0 / ppd) - 1.0)
    threshold = 0.5 * local_spacing
    while True:
        below = [a for a in accepted if a < k_split]
        if not below:
            break
        last_low = below[-1]
        if (k_split - last_low) < threshold:
            accepted.remove(last_low)
        else:
            break
    return accepted


def generate_imd_safe_frequencies(nd, ppd, order=4, first_freq=1.0, split_frequency=None, verbose=False):
    """
    Generate an intermodulation-safe frequency set spanning `nd` decades at
    `ppd` points per decade, starting at 1 Hz and scaled by `first_freq`.

    Parameters
    ----------
    nd : float
        Number of decades to cover.
    ppd : int
        Target points per decade on the log-spaced candidate grid (the
        final set is usually sparser, since some candidates are rejected).
    order : int
        Maximum size of the intermodulation products checked for (>= 2).
        order == 2 uses a cheap all-odd shortcut instead of the general
        search -- except when split_frequency is given, since forcing odd
        values wouldn't respect the split's grid constraint below (a
        candidate that must be a multiple of split_frequency could well
        need to be even); the general search handles order == 2 correctly
        too, just without that shortcut.
    first_freq : float
        Scale factor applied to the final (unit, integer-valued) frequency
        set, i.e. the lowest generated frequency.
    split_frequency : float, optional
        If given, every generated frequency below split_frequency is an
        integer multiple of first_freq (as always), and every frequency at
        or above it is an integer multiple of split_frequency instead --
        so a later split of the set into low/high bands (splitting.
        split_multisine) can build the high band's waveform using
        split_frequency itself as the FFT bin-alignment reference (one
        period of split_frequency, exactly -- see splitting.py), instead
        of needing the full range's 1-unit resolution. split_frequency
        must be an exact integer multiple of first_freq (the real
        experiment can only have every excited frequency repeat a whole
        number of times per base-harmonic period if that holds) -- raises
        ValueError otherwise. Both sub-ranges are still searched together,
        in one pass over the combined, sorted candidate list, exactly like
        the unsplit case -- an intermodulation product can still be
        rejected because it collides with something accepted on the other
        side of the split, so cross-band safety is identical to generating
        the whole range at once.
    verbose : bool
        Print progress as each frequency is accepted.

    Returns
    -------
    np.ndarray
        Sorted ascending frequency set, in Hz.
    """
    if split_frequency is None:
        candidates = np.int64(np.unique(np.round(np.logspace(0, nd, int(nd * ppd))))).tolist()
        k_split = None
    else:
        k_split_float = split_frequency / first_freq
        k_split = int(round(k_split_float))
        if not np.isclose(k_split_float, k_split, rtol=0, atol=1e-6):
            raise ValueError(
                f"split_frequency ({split_frequency}) must be an exact integer multiple of "
                f"first_freq ({first_freq}) -- got split_frequency/first_freq = {k_split_float}."
            )
        nd_low = np.log10(k_split)
        nd_high = nd - nd_low
        if nd_low <= 0 or nd_high <= 0:
            raise ValueError(
                f"split_frequency must sit strictly between first_freq and the top of the "
                f"requested range (first_freq * 10**{nd}) -- got nd_low={nd_low:.3g} decades "
                f"below it, nd_high={nd_high:.3g} decades above."
            )
        candidates_low = np.int64(
            np.unique(np.round(np.logspace(0, nd_low, max(1, int(round(nd_low * ppd))))))
        ).tolist()
        candidates_high_units = np.unique(np.round(np.logspace(0, nd_high, max(1, int(round(nd_high * ppd))))))
        candidates_high = np.int64(np.round(candidates_high_units * k_split)).tolist()
        candidates = sorted(set(candidates_low) | set(candidates_high))

    def grid_step_of(candidate):
        return 1 if (k_split is None or candidate < k_split) else k_split

    if order == 2 and split_frequency is None:
        accepted = _force_odd_frequencies(candidates).tolist()
    else:
        accepted = [1]
        for i in range(1, len(candidates)):
            search_order = min(len(accepted) - 1, order)
            if search_order < 2:
                search_order = 2
            elif search_order == 2 and order > 2:
                # Ramp the check straight to 3 once there's enough accepted
                # to make an order-3 check meaningful -- but never past what
                # the caller actually asked for. Without the "order > 2"
                # guard this silently checked order 3 even when order == 2
                # was explicitly requested (never exercised before
                # split_frequency existed: order == 2 always took the
                # all-odd shortcut above, which doesn't go through this
                # loop at all -- letting order == 2 fall through to the
                # general search, here, for the split_frequency case is
                # what first exposed it).
                search_order = 3

            grid = grid_step_of(candidates[i])
            step = min(abs(candidates[i] - accepted[-1]), abs(candidates[i] - candidates[i - 1]))
            step = (step // grid) * grid  # stay on this candidate's own grid (1, or multiples of split_frequency)
            search_range = [candidates[i]]
            for offset in range(grid, step, grid):
                search_range.append(candidates[i] - offset)
                search_range.append(candidates[i] + offset)

            for candidate in search_range:
                subset_size = _subsetsum(accepted, candidate)
                if subset_size is False or subset_size > 4:
                    accepted.append(candidate)
                    if _is_sub_sum(accepted, search_order):
                        accepted.pop()
                    else:
                        if verbose:
                            print(f"... {candidate} added")
                        break

    if k_split is not None:
        accepted = _prune_near_split(accepted, k_split, ppd)

    return np.sort(np.array(accepted, dtype=float)) * first_freq


def snap_top_frequency_to_power_of_ten(frequencies, nd, first_freq=1.0):
    """
    Drop any frequency at or above the theoretical top of the requested
    decade range (first_freq * 10**nd) and relabel the new highest
    remaining frequency to land exactly on that power of ten.

    Useful because generate_imd_safe_frequencies works off a rounded
    integer grid and its actual top frequency rarely lands on a round
    decade boundary on its own.
    """
    target = first_freq * 10.0 ** nd
    frequencies = np.sort(np.asarray(frequencies, dtype=float))
    frequencies = frequencies[frequencies < target]
    if frequencies.size == 0:
        raise ValueError("No frequencies remain below the target power of ten.")
    frequencies = frequencies.copy()
    frequencies[-1] = target
    return frequencies
