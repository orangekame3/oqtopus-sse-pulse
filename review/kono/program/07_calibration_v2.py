# This program was modified from the original version provided by Mr. Hirose.


import traceback

# from qubex.pulse import PulseSchedule, FlatTop, Blank
# import numpy as np
# import json

from oqtopus_sse_pulse.libs.kono.calib import calibrate, CustomExperiment

# import time
# import datetime


# 使用するqubit等の設定
chip_id = '64Qv3'
qubits = ['Q36', 'Q37', 'Q38', 'Q42', 'Q43'] # 測定対象のqubitリスト
# qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43'] # 測定対象のqubitリスト
# qubits = ['Q36', 'Q37', 'Q38', 'Q42', 'Q43', 'Q52', 'Q54', 'Q55'] # 測定対象のqubitリスト
# qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q48', 'Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55'] # 測定対象のqubitリスト
params_dir = "/sse/in/repo/kono/params"
calib_note_path = "/sse/in/repo/kono/calib_note.json"


try:
    # CustomExperimentクラスのインスタンスを作成
    ex = CustomExperiment(
        chip_id=chip_id,
        qubits=qubits,
        params_dir=params_dir,
        calib_note_path=calib_note_path
    )
    ex.connect()

    # calibration
    calibrate(ex)


# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()