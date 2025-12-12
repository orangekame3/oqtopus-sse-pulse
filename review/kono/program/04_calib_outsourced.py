import qubex as qx

from oqtopus_sse_pulse.libs.kono.calib import calibrate


# settings; CHANGE HERE DEPENDING ON YOUR ENVIRONMENT
ex = qx.Experiment(
    chip_id="64Qv3",
    muxes=[9],
    params_dir="/sse/in/repo/kono/params",
    calib_note_path="/sse/in/repo/kono/calib_note.json"
)
ex.connect()

# calibration
calibrate(ex)

# WRITE YOUR CODE BELOW
# USE ex OBJECT TO EMPLOY THE CALIBRATION DATA ABOVE