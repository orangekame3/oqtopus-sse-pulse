import traceback
from datetime import datetime

from qubex.experiment import Experiment
from qubex.pulse import FlatTop
import numpy as np
import json

import time

from .dictionary import check_keys
from .classifier import Classifier


def measure_qubit_states(
        qubit_settings: dict = {
            "chip_id": "64Qv3",
            "muxes": [9],
            "qubit": ["Q36"]
        },
        config_file_info: dict = {
            "params_dir": "/sse/in/repo/kono/params",
            "calib_note_path": "/sse/in/repo/kono/calib_note.json"
        }, 
        classifier: dict = {
            "Q36": Classifier()
        },
        num_shots: int = 1,
        delay_time: float = 1.0
    ):

    try:
        # 分類器とキーの確認
        for qubit in qubit_settings["qubit"]:
            if classifier[qubit] is None:
                raise ValueError("Classifier instance must be provided.")
        check_keys(input_dict=qubit_settings, required_keys=["chip_id", "muxes", "qubit"])
        check_keys(input_dict=config_file_info, required_keys=["params_dir", "calib_note_path"])

        print(f"=== {qubit_settings.get('qubit', 'Q36')}: start state measurement ===")
        
        qubit = qubit_settings.get("qubit", ["Q36"])

        # qubexのExperimentクラスのインスタンスを作成
        exp = Experiment(
            chip_id=qubit_settings.get("chip_id", "64Qv3"),
            muxes=qubit_settings.get("muxes", [9]),
            params_dir=config_file_info.get("params_dir", "/sse/in/repo/kono/params"),
            calib_note_path=config_file_info.get("calib_note_path", "/sse/in/repo/kono/calib_note.json")
        )

        # デバイスに接続
        exp.connect()

        calib_note = exp.calib_note
        # Convert CalibrationNote to dict for JSON serialization
        calib_note_dict = calib_note._dict if calib_note else None

        hpi_pulses = []
        for qubit in qubit_settings["qubit"]:
            hpi_amplitude = calib_note_dict["hpi_params"][qubit]["amplitude"]  # calib_note.jsonからhpiパルス振幅を取得
            # hpi_amplitude = calib_note_dict['hpi_amplitude'][qubit]  # calib_note.jsonからhpiパルス振幅を取得

            # hpiパルスオブジェクトを作成
            hpi_pulse = FlatTop(
                        duration = 32,
                        amplitude = hpi_amplitude,
                        tau = 12,
                    )
            hpi_pulses.append(hpi_pulse)

        # 波形シーケンスの辞書を作成
        sequence = {}
        for qubit, hpi_pulse in zip(qubit_settings["qubit"], hpi_pulses):
            sequence[qubit] = hpi_pulse.repeated(2)
        # sequence = {qubit: hpi_pulse.repeated(2)}

        for _ in range(num_shots):
            # 開始時刻を取得
            start_time = time.time()

            # measureメソッドで測定を実行
            res = exp.measure(
                sequence = sequence, # 自作の波形シーケンスを指定
                mode = "avg", # 単発射影測定の場合は"single"を指定
                shots = 1 # ショット数
                # shots = 1024 # ショット数
            ) # MeasureResultクラスを出力する

            # 終了時刻を取得
            end_time = time.time()

            # 結果を整形してJSON形式で出力
            result = {}
            result["state"] = {}
            result["raw_data"] = {}
            for qubit in qubit_settings["qubit"]:
                # result = {
                #     "time_range": (np.arange(len(res.data[qubit].raw)) * 8).tolist(),  # 読み出しのサンプリング間隔は8ns
                #     "raw_data_real": res.data[qubit].raw.real.tolist(),
                #     "raw_data_imag": res.data[qubit].raw.imag.tolist(),
                #     "kerneled_data_real": (res.data[qubit].kerneled.real / len(res.data[qubit].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
                #     "kerneled_data_imag": (res.data[qubit].kerneled.imag / len(res.data[qubit].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
                # }
                kerneled_data_real = (res.data[qubit].kerneled.real / len(res.data[qubit].raw)).tolist()  # kerneledデータは合計値なので, 平均値に変換
                kerneled_data_imag = (res.data[qubit].kerneled.imag / len(res.data[qubit].raw)).tolist()  # kerneledデータは合計値なので, 平均値に変換
                state = classifier[qubit](kerneled_data_real, kerneled_data_imag)
                
                result["state"][qubit] = state
                result["raw_data"][qubit] = {
                    "kerneled_data_real": kerneled_data_real,
                    "kerneled_data_imag": kerneled_data_imag,
                }

            result["time"] = {
                "start_time": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                "end_time": datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                "measurement_time": end_time - start_time
            }
            print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

            # 次の実行までの残り時間を計算
            next_time = start_time + delay_time
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)


    # 例外処理
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()

    # 終了処理
    finally:
        print("end state measurement")