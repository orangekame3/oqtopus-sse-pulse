import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, FlatTop

import numpy as np
from copy import deepcopy

from .dictionary import check_keys


class Classifier:
    def __init__(
        self,
        qubit_settings: dict = {
            "chip_id": "64Qv3",
            "muxes": [9],
            "qubit": "Q36"
        }, 
        config_file_info: dict = {
            "params_dir": "/sse/in/repo/kono/params",
            "calib_note_path": "/sse/in/repo/kono/calib_note.json"
        }

    ):
        try:
            # キーの確認
            check_keys(input_dict=qubit_settings, required_keys=["chip_id", "muxes", "qubit"])
            check_keys(input_dict=config_file_info, required_keys=["params_dir", "calib_note_path"])

            self.qubit_settings = qubit_settings
            self.config_file_info = config_file_info

            print(f"=== {self.qubit_settings.get('qubit', 'Q36')}: Classifier Calibration Start ===")

            result_g = self.measure_g_state()
            result_e = self.measure_e_state()
            # 測定で得られた全データ
            kerneled_data_real_g = result_g["kerneled_data_real"]
            kerneled_data_imag_g = result_g["kerneled_data_imag"]
            x_g = np.mean(kerneled_data_real_g)
            y_g = np.mean(kerneled_data_imag_g)

            kerneled_data_real_e = result_e["kerneled_data_real"]
            kerneled_data_imag_e = result_e["kerneled_data_imag"]
            x_e = np.mean(kerneled_data_real_e)
            y_e = np.mean(kerneled_data_imag_e)

            self.slope = -(x_e - x_g) / (y_e - y_g)
            self.intercept = 1 / 2 * (x_e ** 2 - x_g ** 2 + y_e ** 2 - y_g ** 2) / (y_e - y_g)
            self.sign_e = 1 if y_e - (self.slope * x_e + self.intercept) > 0 else -1

            print(f"=== {self.qubit_settings.get('qubit', 'Q36')}: Classifier Calibration End ===")
        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()


    def __call__(self, real: float, imag: float) -> int:
        """
        IQ平面上の点(real, imag)を入力とし, qubit状態'0'か'1'かを分類する関数.
        """
        if self.sign_e * (imag - (self.slope * real + self.intercept)) > 0:
            return 1
        else:
            return 0
            
    def measure_g_state(self):
        print(f"=== {self.qubit_settings.get('qubit', 'Q36')}: start g state measurement ===")
        try:
            qubit = self.qubit_settings.get("qubit", "Q36")

            # qubexのExperimentクラスのインスタンスを作成
            exp = Experiment(
                chip_id=self.qubit_settings.get("chip_id", "64Qv3"),
                muxes=self.qubit_settings.get("muxes", [9]),
                params_dir=self.config_file_info.get("params_dir", "/sse/in/repo/kono/params"),
                calib_note_path=self.config_file_info.get("calib_note_path", "/sse/in/repo/kono/calib_note.json")
            )

            # デバイスに接続
            exp.connect()

            # 空の波形リストを作成
            waveform = []

            # waveformリストを, qubexのPulseクラスのインスタンスに変換
            waveform = Pulse(waveform)

            # 波形シーケンスの辞書を作成
            sequence = {qubit: waveform}

            # measureメソッドで測定を実行
            res = exp.measure(
                sequence = sequence, # 自作の波形シーケンスを指定
                mode = "single", # 単発射影測定の場合は"single"を指定
                shots = 1024, # ショット数
            ) # MeasureResultクラスを出力する


            # 結果を整形してJSON形式で出力
            result = {
                "kerneled_data_real": res.data[qubit].raw.real.tolist(),
                "kerneled_data_imag": res.data[qubit].raw.imag.tolist(),
            }

            # 結果の出力
            # print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return result

        # 例外処理
        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()

        # 終了処理
        finally:
            print(f"=== {self.qubit_settings.get('qubit', 'Q36')}: end g state measurement ===")

    def measure_e_state(self):
        print(f"=== {self.qubit_settings.get('qubit', 'Q36')}: start e state measurement ===")
        try:
            qubit = self.qubit_settings.get("qubit", "Q36")

            # qubexのExperimentクラスのインスタンスを作成
            exp = Experiment(
                chip_id=self.qubit_settings.get("chip_id", "64Qv3"),
                muxes=self.qubit_settings.get("muxes", [9]),
                params_dir=self.config_file_info.get("params_dir", "/sse/in/repo/kono/params"),
                calib_note_path=self.config_file_info.get("calib_note_path", "/sse/in/repo/kono/calib_note.json")
            )

            # デバイスに接続
            exp.connect()

            calib_note = exp.calib_note
            # Convert CalibrationNote to dict for JSON serialization
            calib_note_dict = calib_note._dict if calib_note else None
            hpi_amplitude = calib_note_dict["hpi_params"][qubit]["amplitude"]  # calib_note.jsonからhpiパルス振幅を取得
            # hpi_amplitude = calib_note_dict['hpi_amplitude'][qubit]  # calib_note.jsonからhpiパルス振幅を取得

            # hpiパルスオブジェクトを作成
            hpi_pulse = FlatTop(
                        duration = 32,
                        amplitude = hpi_amplitude,
                        tau = 12,
                    )

            # 波形シーケンスの辞書を作成
            sequence = {qubit: hpi_pulse.repeated(2)}

            # measureメソッドで測定を実行
            res = exp.measure(
                sequence = sequence, # 自作の波形シーケンスを指定
                mode = "single", # 単発射影測定の場合は"single"を指定
                shots = 1024, # ショット数
            ) # MeasureResultクラスを出力する


            # 結果を整形してJSON形式で出力
            result = {
                "kerneled_data_real": res.data[qubit].raw.real.tolist(),
                "kerneled_data_imag": res.data[qubit].raw.imag.tolist(),
            }

            # 結果の出力
            # print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return result

        # 例外処理
        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()

        # 終了処理
        finally:
            print(f"=== {self.qubit_settings.get('qubit', 'Q36')}: end e state measurement ===")


def initizialize_classifiers(
    qubit_settings: dict,
    config_file_info: dict
) -> dict[str, Classifier]:
    classifiers = {}
    qubits = qubit_settings.get("qubit", ["Q36"])
    for qubit in qubits:
        qubit_settings_temp = deepcopy(qubit_settings)
        qubit_settings_temp["qubit"] = qubit
        clf = Classifier(
            qubit_settings=qubit_settings_temp,
            config_file_info=config_file_info
        )
        classifiers[qubit] = clf
    return classifiers