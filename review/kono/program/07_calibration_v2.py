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
qubits_mux09 = ['Q36', 'Q37', 'Q38', 'Q39'] # MUX09に接続されている測定対象のqubitリスト
qubits_mux10 = ['Q41', 'Q42'] # MUX10に接続されている測定対象のqubitリスト
# qubits_mux10 = ['Q40', 'Q41', 'Q42', 'Q43'] # MUX10に接続されている測定対象のqubitリスト
qubits_mux11 = ['Q44', 'Q47'] # MUX11に接続されている測定対象のqubitリスト
# qubits_mux11 = ['Q44', 'Q45', 'Q46', 'Q47'] # MUX11に接続されている測定対象のqubitリスト
qubits_mux12 = ['Q48', 'Q49', 'Q50', 'Q51'] # MUX12に接続されている測定対象のqubitリスト
qubits_mux13 = ['Q52', 'Q54', 'Q55'] # MUX13に接続されている測定対象のqubitリスト
# qubits_mux13 = ['Q52', 'Q53', 'Q54', 'Q55'] # MUX13に接続されている測定対象のqubitリスト
qubits_mux14 = ['Q56', 'Q57', 'Q58', 'Q59'] # MUX14に接続されている測定対象のqubitリスト
qubits_mux15 = ['Q60', 'Q62', 'Q63'] # MUX15に接続されている測定対象のqubitリスト
# qubits_mux15 = ['Q60', 'Q61', 'Q62', 'Q63'] # MUX15に接続されている測定対象のqubitリスト
qubits = qubits_mux09 + qubits_mux10
# qubits = qubits_mux13 + qubits_mux14
# qubits = qubits_mux11 + qubits_mux15
# qubits = ['Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q48', 'Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55'] # 測定対象のqubitリスト
# qubits = ['Q36', 'Q37', 'Q38', 'Q42', 'Q43', 'Q52', 'Q54', 'Q55'] # 測定対象のqubitリスト
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