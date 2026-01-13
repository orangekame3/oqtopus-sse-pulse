import qubex as qx
import json

from qubex.experiment.mixin.characterization_mixin import CharacterizationMixin, DEFAULT_INTERVAL, DEFAULT_SHOTS, SAMPLING_PERIOD
from typing import Collection, Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray

import pickle
import base64

import traceback


CLASSIFIERS_BASE64 = ""


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
# def calibrate(ex: CustomExperiment, calib_readout: bool = False):
    try:
        # start calibration
        ex.obtain_rabi_params(plot=False)                                                           # Rabi measurement

        # detect errors in Rabi measurement
        err_qubits = []
        for qubit, rabi_params in ex.calib_note._dict.items():
            if rabi_params["frequency"] is None:
                err_qubits.append(qubit)
        if len(err_qubits) > 0:
            raise RuntimeError(f"Rabi measurement failed for qubits: {', '.join(err_qubits)}")

        # continue calibration if Rabi measurement is successful
        control_frequencies = ex.calibrate_control_frequency(plot=False)                            # calibrate qubit frequencies
        ex.modified_frequencies(control_frequencies)                                            # update qubit frequencies
        # control_amplitude = {}
        # for qubit in ex.qubit_labels:
        #     qres = ex.measure_qubit_resonance(target=qubit, plot=False, save_image=False)      # measure qubit resonance
        #     control_amplitude[qubit] = qres["estimated_amplitude"]

        # if calib_readout:
        print("Warning!: just measures readout frequencies, not runs calibration")
        readout_frequencies = ex.calibrate_readout_frequency(targets=ex.qubit_labels)       # calibrate readout frequencies

        ex.calibrate_hpi_pulse(plot=False)                                                      # calibrate hpi pulse
        t1 = ex.t1_experiment(plot=False)                                                       # T1 measurement
        t1 = t1.data                                                                            # store results of T1 measurement
        t2 = ex.t2_experiment(plot=False)                                                       # T2 measurement
        t2 = t2.data                                                                            # store results of T2 measurement
        cls = ex.build_classifier(plot=False)                                                   # build classifiers

        # summarize results
        calib_note = ex.calib_note
        calib_note_dict = calib_note._dict if calib_note else None

        props = {
            "resonator_frequency": {
                key: readout_frequencies[key] for key in readout_frequencies
            },
            "qubit_frequency": control_frequencies,
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

        # params = {
        #     key: control_amplitude[key] for key in control_amplitude
        # }

        # シリアライズ（バイナリ→Base64文字列）
        cls_b = pickle.dumps(ex.classifiers, protocol=pickle.HIGHEST_PROTOCOL)
        cls_text = base64.b64encode(cls_b).decode("utf-8")

        # output
        result: dict = {"calib_note": calib_note_dict, "props": props, "classifiers": cls_text}
        # result: dict = {"calib_note": calib_note_dict, "props": props, "params": params, "classifiers": cls_text}
        print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    # 例外処理
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()


def restore_classifiers_from_base64() -> dict[str, any]:
    # デシリアライズ（Base64文字列→バイナリ→オブジェクト）
    cls_b = base64.b64decode(CLASSIFIERS_BASE64.encode("utf-8"))
    classifiers = pickle.loads(cls_b)
    return classifiers