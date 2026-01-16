import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, FlatTop, Blank
import numpy as np
import json
import time
import datetime
import base64

# 使用するqubitの設定
chip_id='64Qv3'
# muxes=[9,10]
# qubit = ['Q36']
# exclude_qubits = ['Q37','Q38','Q39','Q40','Q41','Q42','Q43']
# qubit = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q43'] # NG
# exclude_qubits = ['Q41','Q42']
# qubit = ['Q43']
# exclude_qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42'] # OK
# qubit = ['Q42']
# exclude_qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41','Q43'] # OK
# qubit = ['Q42', 'Q43']
# exclude_qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41'] # OK
# qubit = ['Q36','Q37','Q38','Q39','Q40']
# exclude_qubits = ['Q41','Q42','Q43']  # 除外するqubitリスト
# qubit = ['Q36','Q37','Q38','Q42','Q43'] # Q40, Q41は使用不可, Q39は実質使用不可
# qubit = ['Q36','Q37','Q38','Q42','Q43','Q48','Q50','Q51','Q52','Q53','Q54','Q55']  # 測定対象qubitリスト MUX12,MUX13追加 Q49, Q53は使用不可
qubits = [
    "Q36", "Q37", "Q38", # mux09 
    "Q42", # mux10
    "Q44", "Q47", # mux11
    "Q52", "Q54", "Q55", # mux13
    "Q56", "Q57", "Q58", # mux14
    "Q60", "Q62", "Q63", # mux15
    ]
# {'err_qubits': ['Q43', 'Q45', 'Q46', 'Q53', 'Q61'], 'status': 'failed'}
# exclude_qubits = ['Q41']  # 除外するqubitリスト
time_idle = 200  # 各piパルス間の待ち時間(ns)
counts_mes = 380000  # 測定回数 380000程度が限界（仕様上は100万回程度）
interval = 150 * 1000 # measure sequence interval (ns)
duration = 32  # readout pulse duration
#num_iterations = 16  # 繰り返し回数を指定
num_iterations = 1  # 繰り返し回数を指定 18程度が限界

print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
#        muxes=muxes,
        qubits = qubits,
#        exclude_qubits = exclude_qubits,
        params_dir="/sse/in/repo/hirose/params", # <-- 自分のparamsディレクトリのパスに変更してください
        calib_note_path="/sse/in/repo/hirose/calib_note.json" # <-- 自分のcalib_noteファイルのパスに変更してください
    )

    # デバイスに接続
    exp.connect()

    # summarize results
    calib_note = exp.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None

    targets = qubits  # 測定対象qubitリスト

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
    
    def sequence(time_idle: int) -> PulseSchedule:
        with PulseSchedule(targets) as ps:
            for target in targets:
#                ps.add(target, hpi_pulse[target])
#                ps.add(target, hpi_pulse[target])
                ps.add(target, pi_pulse[target])
                ps.add(target, Blank(time_idle))  # 指定した待ち時間だけ待機
        return ps

    # 各qubitに対して分類器を作成
    exp.build_classifier(plot=False)
    cls = exp.measurement._classifiers

    # 結果を整形してJSON形式で出力
    result = {}
    result["time_idle"] = time_idle
    result["counts_mes"] = counts_mes
    result["dt"] = 544 + interval + time_idle + duration
    result["measurements"] = []  # 測定結果を格納するリスト

    # 繰り返し測定を実行
    for iteration in range(num_iterations):
        # 測定開始時刻を記録
        start_timestamp = time.time()
        start_datetime = datetime.datetime.fromtimestamp(start_timestamp).isoformat()
        
        # measureメソッドで測定を実行
        res = exp.measure(
            interval = interval,
            sequence = sequence(time_idle), # 自作の波形シーケンスを指定
            mode = "single", # 単発射影測定の場合は"single"を指定
            shots = counts_mes # ショット数
        ) # MeasureResultクラスを出力する

        # 各測定の結果を格納
        measurement_data = {
            "iteration": iteration,
            "start_timestamp": start_timestamp,
            "start_datetime": start_datetime,
        }
        
        for target in targets:
#             measurement_data[target] = {
# #                "data_real": res.data[target].raw.real.tolist(),
# #                "data_imag": res.data[target].raw.imag.tolist(),
#                 "classed_data": cls[target].predict(res.data[target].raw).tolist(),
#             }

            arr = cls[target].predict(res.data[target].raw.reshape(-1))
            packed = np.packbits(arr)
            # バイト列をbase64でエンコード
            packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
            measurement_data[target] = packed_b64


        result["measurements"].append(measurement_data)
#        print(f"Completed iteration {iteration + 1}/{num_iterations} at {start_datetime}")

    # 結果の出力
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()

# 終了処理
finally:
    print("end program")
