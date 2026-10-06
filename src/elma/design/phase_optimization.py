"""
Crest-factor phase optimization for a multisine.Multisine instance.

Multisine.best_random_phases already provides a random-phase search; this
adds the differential-evolution global optimizer from the original
MultisineGenerator.py script's phase_optimizer(method='optimization').

That script's version has a parameter-count bug: its objective function
crest_factor_optimize(t, phases, frequencies, amplitudes) is called with
only 3 positional arguments (via lmfit's `args=(f, a)`), so `t` silently
receives the Parameters object, `phases` receives the frequency array, and
`amplitudes` is left unbound -- it would raise a TypeError if actually
invoked. This is a corrected port: the objective reproduces
Multisine.compute_multisine's own sin-based summation directly, so the
optimized phases apply to the exact waveform convention the rest of the
GUI already uses.

Two things make differential_evolution expensive here and both are
addressed below:

1. Each objective evaluation needs a full waveform. Direct sin-summation
   is O(n_freq * n_points) per evaluation; a differential_evolution
   restart can need population_size * generations evaluations, so for a
   wide multisine (millions of points, dozens of frequencies) that adds
   up fast. Both objectives here build the waveform via
   elma.design.waveform.compute_multisine_fft (an O(N log N)
   IFFT, exact for DEIS multisines since every frequency is by
   construction an integer multiple of the base frequency -- see that
   module's docstring) instead of a manual sin loop.
2. lmfit's differential_evolution wrapper does NOT respect scipy's own
   `maxiter` kwarg (it warns and silently drops it). The knob that
   actually matters is `max_nfev`, which lmfit -- despite the name --
   forwards straight to scipy's DE as its generation count (`maxiter`),
   defaulting to `2000*(n_params+1)` when unset. For a few dozen
   frequencies that's tens of thousands of generations, each evaluating
   an entire population (`popsize * n_params` individuals) -- tens of
   millions of evaluations before fix #1 is even a factor. `max_nfev`/
   `popsize` are set here to defaults sized for interactive GUI use, and
   both remain overridable via **de_kws. This is a fundamental scaling
   limit of population-based global optimizers, not something fix #1
   alone resolves -- for large frequency sets, optimize_phases_random_search
   below scales far better, since its cost is just num_iterations,
   independent of dimensionality.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from multisine import compute_crest_factor

from elma.design.waveform import compute_multisine_fft


def _lmfit():
    """lmfit is only needed for the differential-evolution optimiser: imported on first use so
    that importing elma.design (e.g. just to load a multisine file) does not require it."""
    import lmfit
    return lmfit


def optimize_phases_differential_evolution(
    multisine_obj, num_optimizations=5, seed=None, max_nfev=30, popsize=8, **de_kws
):
    """
    Minimize `multisine_obj`'s crest factor via lmfit's differential
    evolution global optimizer. Since differential_evolution is itself
    stochastic/population-based, the search is repeated `num_optimizations`
    times from independent random phase seeds and the best run is kept --
    mirroring the repeat-and-keep-best pattern of Multisine.best_random_phases.

    Each restart runs a full differential-evolution search, which is much
    more expensive than one random-phase trial -- keep num_optimizations
    small (a handful, not hundreds). `max_nfev` (generations) and `popsize`
    default well below scipy's own defaults to keep this tractable for GUI
    use, especially for wide (many-decade) frequency sets; pass larger
    values (or other **de_kws) for a more thorough but slower search.

    Returns
    -------
    phases : np.ndarray
    crest_factor : float
    """
    rng = np.random.default_rng(seed)
    n_freq = multisine_obj.frequencies.size

    def objective(params):
        phases = np.array([params[f"p{j}"].value for j in range(n_freq)])
        waveform = compute_multisine_fft(multisine_obj, phases)
        return compute_crest_factor(waveform)

    best_phases, best_cf = None, np.inf
    for _ in range(num_optimizations):
        params = _lmfit().Parameters()
        for j in range(n_freq):
            params.add(f"p{j}", value=rng.uniform(0, 2 * np.pi), min=0, max=2 * np.pi)
        result = _lmfit().minimize(
            objective, params, method="differential_evolution", max_nfev=max_nfev, popsize=popsize, **de_kws
        )
        cf = objective(result.params)
        if cf < best_cf:
            best_cf = cf
            best_phases = np.array([result.params[f"p{j}"].value for j in range(n_freq)])

    return best_phases, best_cf


def optimize_phases_random_search(multisine_obj, iterations=200, seed=None, n_workers=None):
    """
    Faster drop-in replacement for Multisine.best_random_phases: the exact
    same algorithm (try `iterations` random phase sets, keep the one with
    the lowest crest factor), sped up two ways:

    1. Each trial's waveform is built via
       elma.design.waveform.compute_multisine_fft (an O(N log N)
       IFFT) instead of best_random_phases' own O(n_freq * N) direct
       sin-summation loop -- exact, not approximate, for DEIS multisines
       (see that module's docstring), and the dominant speedup at scale:
       ~40x for a 7-decade, 100M-sample multisine.
    2. Evaluates trials concurrently on a thread pool on top of that --
       numpy's FFT and ufuncs release the GIL for large arrays, so this
       gets real multi-core speedup, not just I/O-bound overlap. Set
       n_workers=1 to disable (e.g. if memory-constrained: every
       concurrent trial holds its own full-size waveform array, so peak
       memory scales with n_workers).

    Returns
    -------
    phases : np.ndarray
    crest_factor : float
    """
    rng = np.random.default_rng(seed)
    n_freq = multisine_obj.frequencies.size

    # Generate all trial phase sets up front (cheap, single-threaded) so
    # no RNG state is shared/mutated across worker threads.
    all_phases = [rng.uniform(0, 2 * np.pi, n_freq) for _ in range(iterations)]

    def evaluate(phases):
        waveform = compute_multisine_fft(multisine_obj, phases)
        return compute_crest_factor(waveform)

    if n_workers is None:
        n_workers = min(iterations, os.cpu_count() or 1)

    if n_workers <= 1:
        crest_factors = [evaluate(p) for p in all_phases]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            crest_factors = list(ex.map(evaluate, all_phases))

    best_index = int(np.argmin(crest_factors))
    return all_phases[best_index], crest_factors[best_index]
