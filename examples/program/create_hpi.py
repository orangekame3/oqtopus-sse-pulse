import traceback
import json

from qubex.experiment import Experiment

print("start program")
try:
    exp = Experiment(
        chip_id="64Qv3",
        muxes=[9],
        params_dir="/sse/in/repo/miyanaga/params",
        calib_note_path="/sse/in/repo/miyanaga/calib_note.json"
    )
    exp.connect()
    res = exp.calibrate_hpi_pulse(targets="Q36")
    calib_note = exp.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None
    result: dict = {"mode": "", "data": {},"calib_note": calib_note_dict}
    for qubit, data in res.data.items():
        result["data"][qubit] = {
            "raw": {
                "I": data.data.real.tolist(), # type: ignore
                "Q": data.data.imag.tolist(), # type: ignore
            },
            "normalized": {
                "I": data.normalized.real.tolist(), # type: ignore
                "Q": data.normalized.imag.tolist(), # type: ignore
            }
        }
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

except Exception as e:
    print("Exception:", e)
    traceback.print_exc()
finally:
    print("end program")
