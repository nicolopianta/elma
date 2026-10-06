"""
Run a DEIS measurement from an experiment settings file saved by the elma GUI ("Save settings..."
in the Experiment Builder tab), without opening the GUI.

    python 6-headless_from_settings_json.py experiment_settings.json

The settings name the multisine design, the potentiostat, the technique sequence (with the per-step
DEIS flags), the AWG, the oscilloscope and the online analysis -- see elma.config for the format.
Needs the instruments (and the elma[design] extra to read the multisine design).
"""
import json
import sys
import time
from pathlib import Path

from elma.builder import build_run
from elma.config import load_design_from_settings, resolve_configuration
from elma.peis_geis import write_peis_geis_csv
from elma.runs import has_finished

settings_path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiment_settings.json")
settings = json.loads(settings_path.read_text(encoding="utf-8"))

multisine, multisine_split, _ = load_design_from_settings(settings)
config = resolve_configuration(settings, multisine, multisine_split)

# what the GUI would show in a dialog before starting
for problem in (config["awg"]["amplitude_error"], config["oscilloscope"]["current_conversion_factor_error"]):
    if problem:
        raise SystemExit(problem)

run = build_run(config, multisine, multisine_split)   # connects to the instruments
run.start()
try:
    while not has_finished(run):
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
    run.stop()

# PEIS/GEIS steps deliver their impedance points on the potentiostat channel
points = getattr(run, "potentiostat", None) and run.potentiostat.peis_geis_points()
if points is not None:
    folder = Path(config["experiment"]["saving_directory"]) / config["experiment"]["experiment_name"]
    write_peis_geis_csv(folder / "peis_geis_results.csv", *points)
