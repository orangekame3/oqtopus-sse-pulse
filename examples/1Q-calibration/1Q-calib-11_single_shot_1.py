import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, FlatTop
import numpy as np
import json

# 使用するqubitの設定
chip_id='64Qv3'
# muxes=[9]
# qubit = 'Q36'
# qubit_frequency = 7.995820  # <-- ここを適切なqubit共鳴周波数に変更してください
# hpi_amplitude = 0.05  # <-- ここを適切なhpi振幅に変更してください

muxes=[1]
qubit = 'Q04'
qubit_frequency = 7.984325
hpi_amplitude = 0.042883  


print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
        muxes=muxes,
    )

    # デバイスに接続
    exp.connect()

    # 一時的に駆動周波数をqubit共鳴周波数に設定
    with exp.modified_frequencies({qubit: qubit_frequency}): 

        # hpiパルスオブジェクトを作成
        hpi_pulse = FlatTop(
                    duration = 32,
                    amplitude = hpi_amplitude,
                    tau = 12,
                )

        # 波形シーケンスの辞書を作成
        sequence = {qubit: hpi_pulse.repeated(2)}

        # measureメソッドで測定を実行
        res = exp.measure(
            sequence = sequence, # 自作の波形シーケンスを指定
            mode = "avg", # 単発射影測定の場合は"single"を指定
            shots = 1 # ショット数
        ) # MeasureResultクラスを出力する


    # 結果を整形してJSON形式で出力
    result = {
        "time_range": (np.arange(len(res.data[qubit].raw)) * 8).tolist(),  # 読み出しのサンプリング間隔は8ns
        "raw_data_real": res.data[qubit].raw.real.tolist(),
        "raw_data_imag": res.data[qubit].raw.imag.tolist(),
        "kerneled_data_real": (res.data[qubit].kerneled.real / len(res.data[qubit].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
        "kerneled_data_imag": (res.data[qubit].kerneled.imag / len(res.data[qubit].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
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
