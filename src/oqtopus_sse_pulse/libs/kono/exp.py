import traceback
from datetime import datetime

from qubex.experiment import Experiment
from qubex.pulse import FlatTop
import numpy as np
import json

import time
# from concurrent.futures import ThreadPoolExecutor, process

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
        sequence = {qubit: hpi_pulse.repeated(2), "Q37": hpi_pulse.repeated(2)}
        # sequence = {qubit: hpi_pulse.repeated(2)}

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
        result_Q37 = {
            "time_range": (np.arange(len(res.data["Q37"].raw)) * 8).tolist(),  # 読み出しのサンプリング間隔は8ns
            "raw_data_real": res.data["Q37"].raw.real.tolist(),
            "raw_data_imag": res.data["Q37"].raw.imag.tolist(),
            "kerneled_data_real": (res.data["Q37"].kerneled.real / len(res.data["Q37"].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
            "kerneled_data_imag": (res.data["Q37"].kerneled.imag / len(res.data["Q37"].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
        }

        # 結果の出力
        state = classifier(result["kerneled_data_real"], result["kerneled_data_imag"])
        result = {
            f"measured_state_{qubit}": state,
            f"kerneled_data_real_{qubit}": result["kerneled_data_real"],
            f"kerneled_data_imag_{qubit}": result["kerneled_data_imag"],
            f"measurement_time_sec_{qubit}": end_time - start_time,
            # 例: usレベルで表示する
            f"start_time_{qubit}": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
            "measured_state_Q37": classifier(result_Q37["kerneled_data_real"], result_Q37["kerneled_data_imag"]),
            "kerneled_data_real_Q37": result_Q37["kerneled_data_real"],
            "kerneled_data_imag_Q37": result_Q37["kerneled_data_imag"],
            "measurement_time_sec_Q37": end_time - start_time,
            # 例: usレベルで表示する
            "start_time_Q37": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
        }
        print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        # return result


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
    # with ThreadPoolExecutor() as executor:
    #     results = list(
    #         executor.map(
    #             measure_state_single_qubit, 
    #             [info.get("classifier") for info in qubit_info],
    #             [info.get("qubit_settings") for info in qubit_info],
    #             [info.get("config_file_info") for info in qubit_info]
    #         )
    #     )
    
    # for result in results:
    #     print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))