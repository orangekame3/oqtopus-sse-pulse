import qubex as qx
import json

from qubex.experiment.mixin.characterization_mixin import CharacterizationMixin, DEFAULT_INTERVAL, DEFAULT_SHOTS, SAMPLING_PERIOD
from typing import Collection, Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubex.pulse import Pulse, PulseSchedule, FlatTop, Blank

targets = ['Q36','Q37','Q38','Q39','Q40','Q43']

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
    ex.calibrate_pi_pulse(plot=False)                                   # calibrate pi pulse
    t1 = ex.t1_experiment(plot=False)                                   # T1 measurement
    t1 = t1.data                                                        # store results of T1 measurement
    t2 = ex.t2_experiment(plot=False)                                   # T2 measurement
    t2 = t2.data                                                        # store results of T2 measurement
    cls = ex.build_classifier(plot=False)                               # build classifiers

    # summarize results
    calib_note = ex.calib_note
    calib_note_dict = calib_note._dict if calib_note else None

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

    raw_data: dict = {
        # "t1": {
        #     key: {
        #         "data": t1[key].data, 
        #         "sweep_range": t1[key].sweep_range,
        #     } for key in t1
        # },
        # "t2": {
        #     key: {
        #         "data": t2[key].data, 
        #         "sweep_range": t2[key].sweep_range,
        #     } for key in t2
        # },
        # "classifiers": cls["data"]                                             # raw data used for building classifiers
    }

    # output
    result: dict = {"calib_note": calib_note_dict, "props": props, "raw_data": raw_data}
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


class CustomExperiment2(CustomCharacterizationMixin, qx.Experiment):
    # inherit the custom calibrate_control_frequency function by extending CustomCharacterizationMixin
#    pass
    def calib_meas(
            self,
            counts: int = DEFAULT_SHOTS,
            interval: int = DEFAULT_INTERVAL,
            idle_time: int = 200,
            duration: int = 32,
        ):
    
        for i in range(5):
            print(f"hello {i}")


        # calibration実行
        calibrate(self)

        # summarize results
        calib_note = self.calib_note
        # Convert CalibrationNote to dict for JSON serialization
        calib_note_dict = calib_note._dict if calib_note else None

        # 各qubitに対応するhpiパルスを作成
        hpi_pulse = {}
        pi_pulse = {}
        for target in targets:
            hpi_pulse[target] = FlatTop(
                                        duration=duration,
                                        amplitude=calib_note_dict['hpi_params'][target]['amplitude'],
                                        tau=12,
                                    )
            pi_pulse[target] = FlatTop(
                                        duration=duration,
                                        amplitude=calib_note_dict['pi_params'][target]['amplitude'],
                                        tau=12,
                                    )

        #sequence 定義
        def sequence(time_idle: int) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    # ps.add(target, hpi_pulse[target])
                    # ps.add(target, hpi_pulse[target])
                    ps.add(target, pi_pulse[target])
                    ps.add(target, Blank(time_idle))  # 指定した待ち時間だけ待機
            return ps
        
        # measureメソッドで測定を実行
        res = self.measure(
            interval = interval,
            sequence = sequence(idle_time), # 自作の波形シーケンスを指定
            mode = "single", # 単発射影測定の場合は"single"を指定
            shots = counts # ショット数
        ) # MeasureResultクラスを出力する

        # 結果を整形してJSON形式で出力0
        result = {}
        result["time_idle"] = idle_time,
        result["counts_mes"] = counts,
        result["time_list"] = (np.arange(len(res.data[target].raw)) * (interval + idle_time + duration)).tolist(),  # 読み出しのサンプリング間隔は8ns
        for target in targets:
            result[target] = {
                "data_real": res.data[target].raw.real.tolist(),
                "data_imag": res.data[target].raw.imag.tolist(),
                "classed_data": cls[target].predict(res.data[target].raw).tolist(),
            }

        result: dict = {"test": 1}
        return result

