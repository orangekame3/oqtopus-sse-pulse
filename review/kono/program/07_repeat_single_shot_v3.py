# This program was modified from the original version provided by Mr. Hirose.


import traceback

from qubex.pulse import PulseSchedule, FlatTop, Blank
import numpy as np
import json

from oqtopus_sse_pulse.libs.kono.calib import calibrate, CustomExperiment

import time
import datetime


# 使用するqubit等の設定
chip_id = '64Qv3'
# muxes = [9, 10]
qubits = ['Q36', 'Q38','Q39','Q40','Q42'] # 測定対象のqubitリスト
# qubits = ['Q36', 'Q38', 'Q39', 'Q40']  # 測定対象のqubitリスト
# exclude_qubits = ['Q37', 'Q41', 'Q42', 'Q43']  # 除外するqubitリスト
time_idle = 1000  # 各hpiパルス間の待ち時間(ns)
counts_mes = 10_000  # 測定回数 (1 ms <=> 10 shots <=> 700 B ~ 1 KB)
params_dir = "/sse/in/repo/kono/params"
calib_note_path = "/sse/in/repo/kono/calib_note.json"
duration = 32  # hpiパルスの全体の長さ(ns)
interval = 100 * 1000  # 測定から初期化までの待機時間(ns単位)
max_execution_time = 1200 * 0.05  # 最大実行時間（秒）

try:
    time_before_calibration = time.time_ns()

    # CustomExperimentクラスのインスタンスを作成
    ex = CustomExperiment(
        chip_id=chip_id,
        # muxes=muxes,
        qubits=qubits,
        # exclude_qubits=exclude_qubits,
        params_dir=params_dir,
        calib_note_path=calib_note_path
    )
    ex.connect()

    # calibration
    # calibrate(ex)
    ex.build_classifier(plot=False)

    time_after_calibration = time.time_ns()
    print(f"Calibration time: {time_after_calibration - time_before_calibration} seconds")

    classifiers = ex.measurement._classifiers

    # classifiers is a dict in the form of: {"Q36": classifier_of_Q36, "Q37": classifier_of_Q37, ...}
    # call classifiers["Q36"].predict(data) to predict the state of qubit Q36 for given data.
    # data should be in the form of NDArray, e.g., np.array([I1 + Q1 j, I2 + Q2 j, ...])

    # WRITE YOUR CODE BELOW
    # USE ex OBJECT TO EMPLOY THE CALIBRATION DATA ABOVE


    # get calib_note
    calib_note = ex.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None

    targets = qubits  # 測定対象qubitリスト

    # 各qubitに対応するhpiパルスを作成
    hpi_pulse = {}
    for target in targets:
        hpi_pulse[target] = FlatTop(
                                    duration=duration,
                                    amplitude=calib_note_dict['hpi_params'][target]['amplitude'],
                                    tau=12,
                                )

    # 波形シーケンスを作成
    def sequence(time_idle: int) -> PulseSchedule:
        with PulseSchedule(targets) as ps:
            for target in targets:
                ps.add(target, hpi_pulse[target])
                ps.add(target, hpi_pulse[target])
                ps.add(target, Blank(time_idle))  # 指定した待ち時間だけ待機
        return ps

    while True:
        time_before_measurement = time.time_ns()
        if (time_before_measurement - time_before_calibration) / 1e9 > max_execution_time:
            break

        # measureメソッドで測定を実行
        res = ex.measure(
            interval = interval,
            sequence = sequence(time_idle), # 自作の波形シーケンスを指定
            mode = "single", # 単発射影測定の場合は"single"を指定
            shots = counts_mes # ショット数
        ) # MeasureResultクラスを出力する


        # 結果を整形してJSON形式で出力
        result = {
            "start_time": time_before_measurement,
            # "time_list": (np.arange(len(res.data[targets[0]].raw)) * (duration * 2 + time_idle + 544 + interval)).tolist(),
            "dt": duration * 2 + time_idle + 544 + interval,
            "states": {}
        }
        for target in targets:
            result["states"][target] = classifiers[target].predict(res.data[target].raw.reshape(-1)).tolist()

        # 結果の出力
        print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()