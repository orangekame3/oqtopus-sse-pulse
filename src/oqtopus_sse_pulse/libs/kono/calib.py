import numpy as np
import qubex as qx

import traceback

from .dictionary import check_qubit_settings, check_config_file_info


def frequency_calibration(qubit_settings: dict, config_file_info: dict):
    # this program is coded referring to qubex:
    # https://github.com/amachino/qubex/blob/develop/docs/examples/experiment/2_frequency_calibration.ipynb

    try:
        # Check keys in the input dictionaries
        check_qubit_settings(qubit_settings)
        check_config_file_info(config_file_info)

        ex = qx.Experiment(
            chip_id=qubit_settings["chip_id"],
            muxes=qubit_settings["muxes"],
            params_dir=config_file_info["params_dir"],
            calib_note_path=config_file_info["calib_note_path"]
        )


        ex.connect()


        result = ex.check_waveform()


        result = ex.obtain_rabi_params()


        # Obtain the Chevron pattern
        result = ex.chevron_pattern(
            ex.qubit_labels,
            detuning_range=np.linspace(-0.05, 0.05, 51),
            time_range=np.arange(0, 201, 4),
        )


        # Update `qubit_frequency` in props.yaml and reload
        ex.reload()


        # Calibrate the control frequency
        ex.calibrate_control_frequency(
            ex.qubit_labels,
            detuning_range=np.linspace(-0.01, 0.01, 21),
            time_range=range(0, 101, 4),
        )


        # Update `qubit_frequency` in props.yaml manually and reload
        ex.reload()

        # Check the Rabi oscillation
        ex.obtain_rabi_params()


        # Calculate the default control amplitudes
        ex.calc_control_amplitudes()

        # Update `control_amplitude` in params.yaml and reload
        ex.reload()


        # Calibrate the readout frequency
        ex.calibrate_readout_frequency(
            ex.qubit_labels,
            detuning_range=np.linspace(-0.01, 0.01, 21),
            time_range=range(0, 101, 4),
        )


        # Update `readout_frequency` in props.yaml manually and reload
        ex.reload()

        # Check the Rabi oscillation
        ex.obtain_rabi_params()
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()