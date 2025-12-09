import traceback

from qubex.experiment import Experiment
from qubex.pulse import FlatTop
import numpy as np
import json

import time

from .dictionary import check_keys
from .classifier import Classifier


def measure_state_single_qubit(
        classifier: Classifier, 
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

    if classifier is None:
        raise ValueError("Classifier instance must be provided.")

    print("start state measurement")

    try:
        # キーの確認
        check_keys(input_dict=qubit_settings, required_keys=["chip_id", "muxes", "qubit"])
        check_keys(input_dict=config_file_info, required_keys=["params_dir", "calib_note_path"])
        
        qubit = qubit_settings.get("qubit", "Q36")

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
        result = {
            "time_range": (np.arange(len(res.data[qubit].raw)) * 8).tolist(),  # 読み出しのサンプリング間隔は8ns
            "raw_data_real": res.data[qubit].raw.real.tolist(),
            "raw_data_imag": res.data[qubit].raw.imag.tolist(),
            "kerneled_data_real": (res.data[qubit].kerneled.real / len(res.data[qubit].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
            "kerneled_data_imag": (res.data[qubit].kerneled.imag / len(res.data[qubit].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
        }

        # 結果の出力
        state = classifier(result["kerneled_data_real"], result["kerneled_data_imag"])
        result = {
            f"measured_state_{qubit}": state,
            f"measurement_time_sec_{qubit}": end_time - start_time,
            f"start_time_{qubit}": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        }
        print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


    # 例外処理
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()

    # 終了処理
    finally:
        print("end state measurement")

def measure_state_multi_qubits(qubit_info: list[dict]):
    if qubit_info is None:
        qubit_settings: dict = {
                "chip_id": "64Qv3",
                "muxes": [9],
                "qubit": "Q36"
            }
        config_file_info: dict = {
            "params_dir": "/sse/in/repo/kono/params",
            "calib_note_path": "/sse/in/repo/kono/calib_note.json"
        }
        classifier = Classifier(
            qubit_settings=qubit_settings,
            config_file_info=config_file_info
        )
        qubit_info = [
            {
                "qubit_settings": qubit_settings,
                "config_file_info": config_file_info,
                "classifier": classifier
            }
        ]
    
    for info in qubit_info:
        measure_state_single_qubit(
            classifier=info.get("classifier"),
            qubit_settings=info.get("qubit_settings"),
            config_file_info=info.get("config_file_info")
        )

    pass