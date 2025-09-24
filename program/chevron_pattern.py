import traceback

from qubex.experiment import Experiment
import numpy as np
import json

print("start program")
try:
    exp = Experiment(
        chip_id="144Qv2",
        muxes=[6],
    )
    exp.connect()
    res = exp.chevron_pattern(targets="Q024")

    result = {
        "time_range": np.array(res["time_range"]).round(6).tolist(),
        "detuning_range": np.array(res["detuning_range"]).round(6).tolist(),
        "frequencies": {k: float(v) for k, v in res["frequencies"].items()},
        "chevron_data": {
            k: np.array(v).round(6).tolist()
            for k, v in res["chevron_data"].items()
        },
        "rabi_rates": {
            k: np.array(v).round(6).tolist()
            for k, v in res["rabi_rates"].items()
        },
        "resonant_frequencies": {
            k: float(v) for k, v in res["resonant_frequencies"].items()
        },
    }

    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()
finally:
    print("end program")
