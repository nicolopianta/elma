import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from pyeclab.api.kbio_tech import ECC_parm, make_ecc_parm, make_ecc_parms
from pyeclab.techniques.functions import reset_duration, set_duration_to_1s
from dataclasses import dataclass, field
from npbuffer import NumpyCircularBuffer

@dataclass(frozen = False)
class ConditionAverage:
    """
    Quantity should have the format of Channel.current_values for example: "Ewe",
    "Ece", "I" or "ElapsedTime". Operator instead can only be ">" or "<".
    Note that using the hardware limits of EC-Lab SDK Ece is not implemented.
    """
    technique_index : int
    quantity : str
    operator : str
    threshold : float
    num_elements : int
    buffer : NumpyCircularBuffer = field(init = False)

    def __post_init__(self):
        # float64: float16 has ~2 mV resolution at 2.6 V and rounds every limit comparison
        self.buffer = NumpyCircularBuffer(self.num_elements, dtype=np.float64)

def condition_avarage_serialization_factory(data):
    result_dict = {}
    for field_name, filed_value in data:
        if field_name == 'buffer' :
            continue
        result_dict[field_name] = filed_value
    return result_dict


def check_software_limits(deischannel):
    """
    Check if a certain averege condition (< or > of a treshold value) is met for a
    value of the sampled data over a certain number of points.
    """
    for condition in deischannel.conditions:
        if deischannel.potentiostat.data_info.TechniqueIndex == condition.technique_index:
          quantity_value = getattr(
               deischannel.potentiostat.current_values,
               condition.quantity,
          ) 
          condition.buffer.push(np.array(quantity_value)) 
          quantity_avarage = np.mean(condition.buffer.get_data())
          if condition.operator == ">" and quantity_avarage >= condition.threshold:
              print(f'{condition.quantity} > {condition.threshold}')
              condition.buffer.empty()
              return True
          elif condition.operator == "<" and quantity_avarage <= condition.threshold:
              print(f'{condition.quantity} < {condition.threshold}')
              condition.buffer.empty()
              return True
    return False


def log_online_analysis_timing(save_dir, message: str):
    """
    Append a timestamped line to <save_dir>/logs/online_analysis_timing.log (no-op when
    save_dir is None; never raises). Records when each block is popped off the scope's
    buffer, how much data was waiting, when calculate() starts/finishes and when a poll
    cycle is skipped because the previous block is still being processed -- enough to work
    out afterwards why a run produced fewer blocks than duration / window period suggests
    (start-up lag before the first full window, the run ending before a last window is
    popped, a slow calculate() eating poll cycles, ...).
    """
    if save_dir is None:
        return
    try:
        log_dir = Path(save_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "online_analysis_timing.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}  {message}\n")
    except Exception:
        pass


def log_online_analysis_error(save_dir):
    """Append the traceback of the exception being handled to
    <save_dir>/logs/online_analysis_errors.log (no-op when save_dir is None; never raises).
    Block calculations run in a bare Thread that swallows exceptions silently, so this log
    is the only place a failed block shows up."""
    if save_dir is None:
        return
    try:
        log_dir = Path(save_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "online_analysis_errors.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}\n")
            f.write(traceback.format_exc())
            f.write("\n")
    except Exception:
        pass
