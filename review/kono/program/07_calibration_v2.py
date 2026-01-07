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
qubits = ['Q36', 'Q38', 'Q40'] # 測定対象のqubitリスト
# qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43'] # 測定対象のqubitリスト
time_idle = 1000  # 各hpiパルス間の待ち時間(ns)
counts_mes = 10_000  # 測定回数 (1 ms <=> 10 shots <=> 700 B ~ 1 KB)
params_dir = "/sse/in/repo/kono/params"
calib_note_path = "/sse/in/repo/kono/calib_note.json"
duration = 32  # hpiパルスの全体の長さ(ns)


try:
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
    calibrate(ex)   # Warning!: just measures readout frequencies, not runs calibration
    # calibrate(ex, calib_readout=True)   # Warning!: just measures readout frequencies, not runs calibration


# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()