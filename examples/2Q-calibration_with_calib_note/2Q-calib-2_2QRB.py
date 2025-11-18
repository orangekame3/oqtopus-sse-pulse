import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, FlatTop, VirtualZ
import numpy as np
import json
from oqtopus_sse_pulse.libs.ogawa.functions_for_RB import *

# 使用するqubitの設定
chip_id='64Qv3'
muxes=[9]

ctrl_qubit = 'Q36'
trgt_qubit = 'Q37'
CR_ch = 'Q36-Q37'
targets = [ctrl_qubit, trgt_qubit, CR_ch]  # 測定対象qubitリスト


# RBのパラメータ設定
n_total_gate_list = np.arange(0, 20, 2)  # 総クリフォード数のリスト
n_repeat = 10  # 各総クリフォード数に対する繰り返し回数
interleaved_gate = None  # Interleaved RB時に挿入するゲート. 通常のRBの場合はNoneを指定.
# interleaved_gate = [
#         ['ZX90'], 
#         [
#             ['II','IX','ZZ','ZY', 'YX','YI','XY','XZ', 'XX','XI','YY','YZ', 'ZI','ZX','IZ','IY'], 
#             [1,1,1,-1, 1,1,1,1, -1,-1,1,1, 1,1,1,-1],
#         ]
#     ] # ZX90の場合.


print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
        muxes=muxes,
        params_dir="/sse/in/repo/ogawa/params", # <-- 自分のparamsディレクトリのパスに変更してください
        calib_note_path="/sse/in/repo/ogawa/calib_note.json" # <-- 自分のcalib_noteファイルのパスに変更してください
    )

    calib_note = exp.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None
    ctrl_hpi_amplitude = calib_note_dict['hpi_amplitude'][ctrl_qubit] # calib_note.jsonからhpiパルス振幅を取得
    trgt_hpi_amplitude = calib_note_dict['hpi_amplitude'][trgt_qubit] # calib_note.jsonからhpiパルス振幅を取得

    # デバイスに接続
    exp.connect()

    RB_dict = {}
    for n_gate in n_total_gate_list:

        RB_dict[str(n_gate)] = {}
        for repeat_idx in range(n_repeat):
    
            rand_clifford_list = make_rand_clifford_list_2Q(
                n_gate = n_gate,
                interleaved_gate = interleaved_gate,
            )

            with PulseSchedule(targets) as ps:
                for clifford in rand_clifford_list:
                    # 逆順にゲートを適用
                    for gate in clifford[0]:
                        if gate == 'XI90':
                            ps.add(ctrl_qubit, hpi_pulse_ctrl) # <-- ctrl_qubit用のhpi_pulse_ctrlを自分で定義してください
                        elif gate == 'ZI90':
                            ps.add(ctrl_qubit, VirtualZ(np.pi/2))
                        elif gate == 'IX90':
                            ps.add(trgt_qubit, hpi_pulse_trgt) # <-- trgt_qubit用のhpi_pulse_trgtを自分で定義してください
                        elif gate == 'IZ90':
                            ps.add(trgt_qubit, VirtualZ(np.pi/2))
                        elif gate == 'ZX90':
                            ps.call(CR_gate_pulse)  # <-- CRゲート用のCR_gate_pulseを自分で定義してください
            
            
            # measureメソッドで測定を実行
            res = exp.measure(
                sequence = ps, 
                mode = "single",
                shots = 1024,
            )

            RB_dict[str(n_gate)][str(repeat_idx)] = {
                "kerneled_single_data_real_ctrl": res.data[ctrl_qubit].raw.real.tolist(),
                "kerneled_single_data_imag_ctrl": res.data[ctrl_qubit].raw.imag.tolist(),
                "kerneled_single_data_real_trgt": res.data[trgt_qubit].raw.real.tolist(),
                "kerneled_single_data_imag_trgt": res.data[trgt_qubit].raw.imag.tolist()
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
