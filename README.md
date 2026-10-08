# ELMA

ELMA (Electrochemistry Lab Multi-frequency Acquisition) drives the lab's hardware setup — Picoscope oscilloscope, TrueForm arbitrary waveform generator, and a BioLogic potentiostat via `pyeclab` — to acquire multi-frequency excitation/response signals from an electrochemical cell in real time, and does the live (online) processing of the streamed data.

Impedance estimation from the acquired (or any other) signals is hardware-agnostic and lives in the separate [`deistools`](https://github.com/federicoscarpioni/DEIStools) package, which ELMA depends on.

## Installation

```
pip install elma            # acquisition + orchestration (needs the instrument libraries)
pip install elma[gui]       # + the graphical interface (elma-gui command)
pip install elma[design]    # + multisine design without the GUI
```

`pip install elma` depends on the `gui-support-v2` branch of the `nicolopianta/DEIStools` fork until the
DMFA/detrending additions are merged upstream (see `pyproject.toml`).

## Three ways to run a measurement

1. **The GUI** — `elma-gui`: *Multisine Designer*, *Experiment Builder* (potentiostat, technique sequence with
   a DEIS on/off flag per step, AWG, oscilloscope, online analysis, PEIS/GEIS, live plots) and *Inspection*
   (load a folder, time-resolved impedance slider, DMFA of the low-frequency band with Fermi-Dirac filters and
   trend removal, post-processing of the impedance over time with Savitzky-Golay / Gaussian / moving average
   or median smoothing and outlier removal, spectra).
2. **From a settings file, without the GUI** — `examples/6-headless_from_settings_json.py` loads the JSON that
   the Experiment Builder saves, resolves it (`elma.config`) and builds the run (`elma.builder`).
3. **By hand** — construct `DEISchannel`, `PicoCalculator`, `BlockCalculator` yourself, as in `examples/1…4`.

## Which steps run DEIS

Every non-Loop step of the technique sequence has a `deis` flag. The AWG multisine is on, and the oscilloscope
data are analysed and saved, only during flagged steps (default: on for CA/CALim/CP/CPLim, off for
OCV/PEIS/GEIS). Build the AWG with `MultisineGenerator.for_steps(...)` and give the same indexes to
`DEISchannel(deis_indexes=...)`; `elma.builder` does both from the sequence.

## Output layout

```
<experiment>/measurement_data.txt            potentiostat time / Ewe / I (EC-Lab's own recording)
<experiment>/metadata.json, metadata_deis_exp.json
<experiment>/pico_aquisition/cycle_<loop>_sequence_<technique>/{voltage,current,impedance}.npy
<experiment>/logs/online_analysis_{timing,errors}.log
<experiment>/peis_geis_results.csv           when the sequence has PEIS/GEIS steps
```

`voltage.npy`/`current.npy` are the decimated signals, `impedance.npy` the online per-block FFT-EIS.
They are rewritten after every block (`BlockCalculator(save_dir=...)`), so a crash or Stop loses nothing.
`deistools.processing.data_loader.load_experiment` reads the folder back.

## Contents

- `elma.deischannel` — orchestrates a full DEIS measurement channel: potentiostat technique, waveform generator, and oscilloscope acquisition together.
- `elma.picocalculator` — streams data from the Picoscope and drives online block-by-block computation.
- `elma.blockcalculator` — live/online processing of streamed voltage/current blocks (uses `deistools.processing` to estimate impedance per block).
- `elma.multisinegen` — multisine waveform generation/sequencing for the AWG.
- `elma.utils` — shared acquisition helpers (software limit conditions, logging, serialization).
- `elma.runs` — run kinds without a `DEISchannel`: potentiostat only, raw whole-run capture, AWG step follower.
- `elma.config` — experiment settings → resolved configuration (capture size, analysis window, AWG amplitude,
  scope current scale, …), no GUI needed.
- `elma.builder` — configuration → run (`build_run`), the code behind the GUI's *Start experiment*.
- `elma.design` — multisine design: IMD-safe frequencies, phase optimisation, band splitting, scaling, JSON.
- `elma.gui` — the graphical interface.

`examples/` contains full measurement scripts showing acquisition + processing wired together against real hardware.

## History note

Split out of [`DEIStools`](https://github.com/federicoscarpioni/DEIStools) `v0.1.0`, which kept acquisition and processing/visualisation in one repo. That combined snapshot remains available at the `DEIStools` repo's `v0.1.0` tag.
