import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, FlatTop, Blank, VirtualZ
import numpy as np
import json
from oqtopus_sse_pulse.libs.ogawa.functions_for_RB import *


# 使用するqubitの設定
chip_id='64Qv3'
muxes=[9]
qubit = 'Q36'

# RBのパラメータ設定
n_total_gate_list = np.arange(0, 1000, 100)  # 総クリフォード数のリスト
n_repeat = 10  # 各総クリフォード数に対する繰り返し回数
# interleaved_gate = None  # Interleaved RB時に挿入するゲート. 通常のRBの場合はNoneを指定.
interleaved_gate = [['X90'], [['I', 'X', 'Z', 'Y'], [1, 1, 1, -1]]] # X90の場合.

print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
        muxes=muxes,
        params_dir="/sse/in/repo/ogawa/params", # <-- 自分のparamsディレクトリのパスに変更してください
        calib_note_path="/sse/in/repo/ogawa/calib_note.json" # <-- 自分のcalib_noteファイルのパスに変更してください
    )

    # デバイスに接続
    exp.connect()

    calib_note = exp.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None
    hpi_amplitude = calib_note_dict['hpi_amplitude'][qubit] # calib_note.jsonからhpiパルス振幅を取得

    targets = [qubit]  # 測定対象qubitリスト
    hpi_pulse = FlatTop(
                        duration = 32,
                        amplitude = hpi_amplitude,
                        tau = 12,
                    )
    
    RB_dict = {}
    for n_gate in n_total_gate_list:

        RB_dict[str(n_gate)] = {}
        for repeat_idx in range(n_repeat):
    
            rand_clifford_list = make_rand_clifford_list_1Q(
                n_gate = n_gate,
                interleaved_gate = interleaved_gate,
            )

            with PulseSchedule(targets) as ps:
                for target in targets:
                    for clifford in rand_clifford_list:
                        # 逆順にゲートを適用
                        for gate in clifford[0]:
                            if gate == 'X90':
                                ps.add(target, hpi_pulse)
                            elif gate == 'Z90':
                                ps.add(target, VirtualZ(np.pi/2))
            
            # sequence = {qubit: hpi_pulse.repeated(n_gate)}
                               
            # measureメソッドで測定を実行
            res = exp.measure(
                sequence = ps, 
                # sequence = sequence,
                mode = "avg", 
                shots = 1024,
            )

            data_real = res.data[qubit].kerneled.real.tolist()
            data_imag = res.data[qubit].kerneled.imag.tolist()

            RB_dict[str(n_gate)][str(repeat_idx)] = {
                "data_real": data_real,
                "data_imag": data_imag,
            }
    

    # 結果を整形してJSON形式で出力
    result = {
        "n_total_gate_list": n_total_gate_list.tolist(),
        "n_repeat": n_repeat,
        "RB_data": RB_dict,
    }

    # 結果の出力
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()

# 終了処理
finally:
    print("end program")
