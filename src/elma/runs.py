"""
Run kinds that are not a DEISchannel. All of them (and DEISchannel) expose start() and
stop(), and either `running` (True while active) or `finished`, so a caller can treat them
alike -- see has_finished().

- PotentiostatOnlyRun: potentiostat (+ optional AWG) without an oscilloscope; relies on EC-Lab's own
  recording (Record_every_dt/dI/dE of each technique).
- RawCaptureRun: potentiostat + AWG + oscilloscope capturing the whole run in one shot, raw
  voltage/current saved at the end; no online processing.
- AWGStepFollower: keeps an AWG in step with the potentiostat's sequence for the run kinds that
  do not go through DEISchannel (which does it on every technique change).
"""
import time
import traceback
from pathlib import Path
from threading import Thread

import numpy as np

from elma.utils import check_software_limits


def has_finished(run) -> bool:
    """True once a run has ended by itself (sequence completed, a limit ended the last step,
    ...) rather than through stop(). RawCaptureRun exposes `finished` (set once the capture is
    saved); DEISchannel and PotentiostatOnlyRun expose `running`."""
    return run.finished if hasattr(run, "finished") else not getattr(run, "running", True)


class AWGStepFollower:
    """
    Polls the channel's current technique index and re-applies `awg.update(index)` whenever it
    changes, which switches the multisine on for the steps in the generator's `sequence_indexes`
    (DEIS-flagged steps) and off for the others.
    """

    def __init__(self, awg, channel, poll_interval=0.2):
        self.awg = awg
        self.channel = channel
        self.poll_interval = poll_interval
        self._stopping = False
        self._thread = None

    def start(self):
        self.awg.update(0)
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        applied = 0
        while not self._stopping:
            index = getattr(self.channel, "current_tech_index", applied)
            if index != applied:
                try:
                    self.awg.update(index)
                except Exception:
                    traceback.print_exc()
                applied = index
            time.sleep(self.poll_interval)

    def stop(self):
        self._stopping = True
        if self._thread is not None:
            self._thread.join(timeout=2)


class _SoftwareLimitsMixin:
    """Software rolling-average limits (ConditionAverage) for runs without DEISchannel:
    check_software_limits() only needs `.conditions` and `.potentiostat`, so it is reused from
    a dedicated 1 Hz polling thread (DEISchannel._run polls at the same rate, and
    ConditionAverage.num_elements is a sample count meant to read as seconds)."""

    def _start_conditions_thread(self):
        if self.conditions:
            self._conditions_thread = Thread(target=self._conditions_loop, daemon=True)
            self._conditions_thread.start()

    def _conditions_loop(self):
        while not self._stopping:
            try:
                if check_software_limits(self):
                    print("Software limit met!")
                    self.channel.end_technique()
            except (AttributeError, TypeError):
                pass  # live data not populated yet right after start() -- retry next tick
            time.sleep(1)


class RawCaptureRun(_SoftwareLimitsMixin):
    """
    Minimal potentiostat + AWG + oscilloscope orchestration for when online decimation/FFT-EIS is
    switched off. Unlike DEISchannel/PicoCalculator (built around per-window online processing)
    it captures the whole run in one shot and saves raw_voltage.npy / raw_current.npy in
    `saving_dir`. With `awg` given, the AWG follows the technique sequence (AWGStepFollower),
    otherwise the AWG channels are simply turned on for the whole run.
    """

    def __init__(self, channel, pico, awg_channels, saving_dir, conditions=(), awg=None):
        self.channel = channel
        self._awg_follower = AWGStepFollower(awg, channel) if awg is not None else None
        self.potentiostat = channel  # alias: check_software_limits() expects this name
        self.pico = pico
        self.awg_channels = awg_channels
        self.saving_dir = Path(saving_dir)
        self.conditions = list(conditions)
        self._capture_thread = None
        self._conditions_thread = None
        self._stopping = False
        self.finished = False

    def start(self):
        if self._awg_follower is not None:
            self._awg_follower.start()
        else:
            for awg in self.awg_channels:
                awg.turn_on()
        self.channel.start()
        self.pico.run_streaming_non_blocking(autoStop=True)
        self._capture_thread = Thread(target=self._wait_and_save, daemon=True)
        self._capture_thread.start()
        self._start_conditions_thread()

    def _wait_and_save(self):
        while not self.pico.autoStopOuter:
            time.sleep(0.5)
        voltage, current = self.pico.get_all_signals()
        self.saving_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.saving_dir / "raw_voltage.npy", voltage)
        np.save(self.saving_dir / "raw_current.npy", current)
        self.finished = True

    def stop(self):
        self._stopping = True
        if getattr(self.channel, "running", False):
            self.channel.stop()
        if self._awg_follower is not None:
            self._awg_follower.stop()
        for awg in self.awg_channels:
            awg.turn_off()
        self.pico.stop()  # also flips autoStopOuter, releasing the wait thread
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=10)
        if self._conditions_thread is not None:
            self._conditions_thread.join(timeout=2)
        self.pico.disconnect()


class PotentiostatOnlyRun(_SoftwareLimitsMixin):
    """
    Potentiostat-only orchestration for when the oscilloscope is switched off entirely: no
    scope, no capture size, no I-range requirement for the scope. The AWG can still be used and
    applies external control as usual (following the technique sequence when `awg` is given),
    but without the scope there is no capture of the perturbation -- only EC-Lab's own, much
    coarser recording.
    """

    def __init__(self, channel, awg_channels, conditions=(), awg=None):
        self.channel = channel
        self._awg_follower = AWGStepFollower(awg, channel) if awg is not None else None
        self.potentiostat = channel  # alias: check_software_limits() expects this name
        self.awg_channels = awg_channels
        self.conditions = list(conditions)
        self._conditions_thread = None
        self._stopping = False

    @property
    def running(self):
        return getattr(self.channel, "running", False)

    def start(self):
        if self._awg_follower is not None:
            self._awg_follower.start()
        else:
            for awg in self.awg_channels:
                awg.turn_on()
        self.channel.start()
        self._start_conditions_thread()

    def stop(self):
        self._stopping = True
        if getattr(self.channel, "running", False):
            self.channel.stop()
        if self._awg_follower is not None:
            self._awg_follower.stop()
        for awg in self.awg_channels:
            awg.turn_off()
        if self._conditions_thread is not None:
            self._conditions_thread.join(timeout=2)
