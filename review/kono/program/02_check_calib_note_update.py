import qubex as qx

import json
import time


# first ex: run obtain_rabi_params and get calib_note
ex = qx.Experiment(
    chip_id="64Qv3",
    muxes=[9],
    params_dir="/sse/in/repo/kono/params",
    calib_note_path="/sse/in/repo/kono/calib_note.json"
)
ex.connect()

ex.obtain_rabi_params(plot=False)
ex.reload()
calib_note = ex.calib_note
# Convert CalibrationNote to dict for JSON serialization
calib_note_dict = calib_note._dict if calib_note else None
result: dict = {"calib_note": calib_note_dict}
print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


# second ex: just load calib_note and output after waiting for 2 minutes
time.sleep(2 * 60)
ex = qx.Experiment(
    chip_id="64Qv3",
    muxes=[9],
    params_dir="/sse/in/repo/kono/params",
    calib_note_path="/sse/in/repo/kono/calib_note.json"
)
ex.connect()
calib_note = ex.calib_note
calib_note_dict = calib_note._dict if calib_note else None
result: dict = {"calib_note": calib_note_dict}
print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))