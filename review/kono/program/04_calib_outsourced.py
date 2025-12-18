import qubex as qx
import json
import numpy as np

from oqtopus_sse_pulse.libs.kono.calib import calibrate, CustomExperiment


# settings; CHANGE HERE DEPENDING ON YOUR ENVIRONMENT
ex = CustomExperiment(
    chip_id="64Qv3",
    muxes=[9],
    params_dir="/sse/in/repo/kono/params",
    calib_note_path="/sse/in/repo/kono/calib_note.json"
)
ex.connect()

# calibration
calibrate(ex)
classifiers = ex.measurement._classifiers

# classifiers is a dict in the form of: {"Q36": classifier_of_Q36, "Q37": classifier_of_Q37, ...}
# call classifiers["Q36"].predict(data) to predict the state of qubit Q36 for given data.
# data should be in the form of NDArray, e.g., np.array([I1 + Q1 j, I2 + Q2 j, ...])

# WRITE YOUR CODE BELOW
# USE ex OBJECT TO EMPLOY THE CALIBRATION DATA ABOVE

result: dict = {"res": classifiers["Q37"].predict(np.array([0.8340454683450931 -0.39310989525805523j, -3.6874393049764302 + 0.038979713714916264j])).tolist()}
print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))