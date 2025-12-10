from oqtopus_sse_pulse.libs.kono.classifier import Classifier, initizialize_classifiers
from oqtopus_sse_pulse.libs.kono.exp import measure_qubit_states
from oqtopus_sse_pulse.libs.kono.calib import frequency_calibration


# initialize parameters
QUBIT_SETTINGS = {
    "chip_id": "64Qv3",
    "muxes": [9],
    "qubit": ["Q36", "Q37"]
    }
CONFIG_FILE_INFO = {
    "params_dir": "/sse/in/repo/kono/params",
    "calib_note_path": "/sse/in/repo/kono/calib_note.json"
}
NUM_SHOTS = 10
DELAY_TIME = 1.0


# initialize classifiers
classifiers = initizialize_classifiers(
    qubit_settings=QUBIT_SETTINGS,
    config_file_info=CONFIG_FILE_INFO
)


# # measure qubit state repetitionally
# measure_qubit_states(
#     qubit_settings=QUBIT_SETTINGS,
#     config_file_info=CONFIG_FILE_INFO,
#     classifier=classifiers,
#     num_shots=NUM_SHOTS,
#     delay_time=DELAY_TIME
# )

frequency_calibration(
    qubit_settings=QUBIT_SETTINGS,
    config_file_info=CONFIG_FILE_INFO
)