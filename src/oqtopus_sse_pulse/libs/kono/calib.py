import qubex as qx
import json

from qubex.experiment.mixin.characterization_mixin import CharacterizationMixin, DEFAULT_INTERVAL, DEFAULT_SHOTS, SAMPLING_PERIOD
from typing import Collection, Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray


class CustomCharacterizationMixin(CharacterizationMixin):
    # define the custom version of calibrate_control_frequency()
    def calibrate_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        result = self.chevron_pattern(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            frequencies=frequencies,
            amplitudes=amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=False,   # do not save image
        )
        resonant_frequencies = result["resonant_frequencies"]

        print("\nResults\n-------")
        print("ge frequency (GHz):")
        for target, frequency in resonant_frequencies.items():
            print(f"    {target}: {frequency:.6f}")
        return resonant_frequencies


class CustomExperiment(CustomCharacterizationMixin, qx.Experiment):
    # inherit the custom calibrate_control_frequency function by extending CustomCharacterizationMixin
    pass


def calibrate(ex: CustomExperiment):
    # start calibration
    ex.obtain_rabi_params(plot=False)                                   # Rabi measurement
    control_frequencies = ex.calibrate_control_frequency(plot=False)    # calibrate qubit frequencies
    ex.modified_frequencies(control_frequencies)                        # update qubit frequencies
    ex.calibrate_hpi_pulse(plot=False)                                  # calibrate hpi pulse
    t1 = ex.t1_experiment(plot=False)                                   # T1 measurement
    t1 = t1.data                                                        # store results of T1 measurement
    t2 = ex.t2_experiment(plot=False)                                   # T2 measurement
    t2 = t2.data                                                        # store results of T2 measurement
    cls = ex.build_classifier(plot=False)

    # summarize results
    calib_note = ex.calib_note
    calib_note_dict = calib_note._dict if calib_note else None
    result: dict = {"calib_note": calib_note_dict}

    props = {
        "qubit_frequencies": control_frequencies,
        "t1": {
            key: t1[key].t1 for key in t1
        }, 
        "t1_err": {
            key: t1[key].t1_err for key in t1
        },
        "t1_r2": {
            key: t1[key].r2 for key in t1
        },
        "t2": {
            key: t2[key].t2 for key in t2
        },
        "t2_err": {
            key: t2[key].t2_err for key in t2
        },
        "t2_r2": {
            key: t2[key].r2 for key in t2
        },
        "readout_fidelities": cls["readout_fidelities"],
        "average_readout_fidelity": cls["average_readout_fidelity"],
    }
    result["props"] = props

    raw_data: dict = {
        "t1": {
            key: {
                "data": t1[key].data, 
                "sweep_range": t1[key].sweep_range,
            } for key in t1
        },
        "t2": {
            key: {
                "data": t2[key].data, 
                "sweep_range": t2[key].sweep_range,
            } for key in t2
        },
        "classifiers": cls["data"]                                             # raw data used for building classifiers
    }
    result["raw_data"] = raw_data

    # output
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    return cls["classifiers"]