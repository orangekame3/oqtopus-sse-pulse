import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, FlatTop
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
        params_dir="/sse/in/repo/ogawa/params", # <-- 自分のparamsディレクトリのパスに変更してください
        calib_note_path="/sse/in/repo/ogawa/calib_note.json" # <-- 自分のcalib_noteファイルのパスに変更してください
    )

    calib_note = exp.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None
    hpi_amplitude = calib_note_dict['hpi_amplitude'][qubit]  # calib_note.jsonからhpiパルス振幅を取得

    # デバイスに接続
    exp.connect()

    targets = [qubit]  # 測定対象qubitリスト
    n_repeat_list = np.arange(21)  # hpiパルスの繰り返し回数の掃引リスト

    # PulseScheduleクラスのhpi_repeatインスタンスを作成. 
    # 1つ引数が必要な関数のオブジェクト. 
    def hpi_repeat(n_repeat: int) -> PulseSchedule:
        with PulseSchedule(targets) as ps:
            for target in targets:
                ps.add(
                    target, # qubitラベル
                    FlatTop(
                        duration = 32,
                        amplitude = hpi_amplitude,
                        tau = 12,
                    ).repeated(n_repeat),  # 繰り返し回数を指定
                )
        return ps

    # 掃引が必要な実験では、sweep_parameterメソッドを使用するのが便利.
    res = exp.sweep_parameter(
        sequence = hpi_repeat, # 引数に関数を指定
        sweep_range = n_repeat_list, # 掃引振幅リスト
    )


    # 結果を整形してJSON形式で出力
    result = {
        "n_repeat_list": res.data[qubit].sweep_range.tolist(),
        "data_real": res.data[qubit].data.real.tolist(),
        "data_imag": res.data[qubit].data.imag.tolist(),
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
