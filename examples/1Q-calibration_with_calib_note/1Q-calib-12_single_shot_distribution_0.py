import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse
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

    # デバイスに接続
    exp.connect()

    # 空の波形リストを作成
    waveform = []

    # waveformリストを, qubexのPulseクラスのインスタンスに変換
    waveform = Pulse(waveform)

    # 波形シーケンスの辞書を作成
    sequence = {qubit: waveform}

    # measureメソッドで測定を実行
    res = exp.measure(
        sequence = sequence, # 自作の波形シーケンスを指定
        mode = "single", # 単発射影測定の場合は"single"を指定
        shots = 1024, # ショット数
    ) # MeasureResultクラスを出力する


    # 結果を整形してJSON形式で出力
    result = {
        "kerneled_data_real": res.data[qubit].raw.real.tolist(),
        "kerneled_data_imag": res.data[qubit].raw.imag.tolist(),
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
