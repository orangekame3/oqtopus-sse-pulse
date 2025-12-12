import json
import qubex as qx


# settings; CHANGE HERE ACCORDING TO YOUR ENVIRONMENT
ex = qx.Experiment(
    chip_id="64Qv3",
    muxes=[9],
    params_dir="/sse/in/repo/kono/params",
    calib_note_path="/sse/in/repo/kono/calib_note.json"
)
ex.connect()

# calibration
ex.obtain_rabi_params(plot=False)
ex.calibrate_hpi_pulse(plot=False)
ex.t1_experiment(plot=False)
ex.t2_experiment(plot=False)
ex.build_classifier(plot=False)
calib_note = ex.calib_note
calib_note_dict = calib_note._dict if calib_note else None
result: dict = {"calib_note": calib_note_dict}
print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


# WRITE YOUR CODE BELOW
# USE ex OBJECT TO EMPLOY THE CALIBRATION DATA ABOVE