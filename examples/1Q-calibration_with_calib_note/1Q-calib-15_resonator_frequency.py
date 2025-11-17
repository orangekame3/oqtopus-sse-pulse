import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule
import numpy as np
import json

# 使用するqubitの設定
chip_id='64Qv3'
muxes=[9]
qubit = 'Q36'

print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
        muxes=muxes,
        params_dir="/sse/in/repo/ogawa/params",
        calib_note_path="/sse/in/repo/ogawa/calib_note.json"
    )

    # デバイスに接続
    exp.connect()

    # ラビ振動測定
    targets = [qubit]  # 測定対象qubitリスト
    time_range = np.arange(0, 200, 4) # 掃引時間リスト (単位: ns, 2nsより細かくはできない)
    ampl = 0.05 # パルス振幅(0~1の範囲の無次元相対量)

    # PulseScheduleクラスのrabi_sequenceインスタンスを作成. 
    # 1つ引数が必要な関数のオブジェクト. 
    def rabi_sequence(T: int) -> PulseSchedule:
        with PulseSchedule(targets) as ps:
            for target in targets:
                ps.add(target, Pulse([ampl] * int(T/2)))  # 長さT nsの矩形波パルス (2nsサンプリングなので2で割っている)
        return ps

    resonator_frequency_list = 10.1975 + np.linspace(-0.1, 0.1, 31)  # 読み出し周波数掃引リスト (GHz)
    result_dict = {}

    for resonator_frequency in resonator_frequency_list:

        # 掃引が必要な実験では、sweep_parameterメソッドを使用するのが便利.
        res = exp.sweep_parameter(
            sequence = rabi_sequence, # 引数に関数を指定
            sweep_range = time_range, # 掃引時間リスト
            frequencies={'R' + qubit: resonator_frequency},
        )

        # 結果を整形してJSON形式で出力
        result = {
            "time_range": res.data[qubit].sweep_range.tolist(),
            "data_real": res.data[qubit].data.real.tolist(),
            "data_imag": res.data[qubit].data.imag.tolist(),
        }

        result_dict[f"{resonator_frequency}"] = result

    result_dict["resonator_frequency_list"] = resonator_frequency_list.tolist()

    # 結果の出力
    print("payload=" + json.dumps(result_dict, ensure_ascii=False, separators=(",", ":")))

# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()

# 終了処理
finally:
    print("end program")
