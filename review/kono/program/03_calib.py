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
t1 = ex.t1_experiment(plot=False).data
t2 = ex.t2_experiment(plot=False).data
ex.build_classifier(plot=False)
calib_note = ex.calib_note
calib_note_dict = calib_note._dict if calib_note else None
result: dict = {
    "calib_note": calib_note_dict,
    "t1": {
        key: {
            "t1": t1[key].t1,
            "t1_error": t1[key].t1_err,
            "r2": t1[key].r2
        } for key in t1
    }, 
    "t2": {
        key: {
            "t2": t2[key].t2,
            "t2_error": t2[key].t2_err,
            "r2": t2[key].r2
        } for key in t2
    }
}
print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

# WRITE YOUR CODE BELOW
# USE ex OBJECT TO EMPLOY THE CALIBRATION DATA ABOVE