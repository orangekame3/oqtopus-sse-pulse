import traceback

from qubex.experiment import Experiment
import numpy as np
import json


print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id='64Qv3',
        muxes=[9],
    )

    # デバイスに接続
    exp.connect()

    # qubexメソッドのchevron_pattern実験を実行
    res = exp.chevron_pattern(
        targets='Q36', # 測定対象qubit
        detuning_range=np.linspace(-0.03, 0.03, 11), # 駆動周波数の掃引範囲 (中心周波数からの相対量). 単位はGHz
        time_range=np.arange(0, 201, 16),  # 測定時間の掃引範囲. 単位はns (最小単位: 2ns)
        # frequencies={'Q36': 8.0},  # 駆動周波数の中心周波数. 未設定の場合は阪大実験チームで登録されている設定値が使用される. 単位はGHz
        # amplitudes={'Q36': 0.1},  # 駆動振幅 (0から1までの無次元相対値). 未設定の場合は阪大実験チームで登録されている設定値が使用される.
        # shots=1024,  # 1点あたりの測定回数. デフォルトは1024
    )

    # 結果を整形してJSON形式で出力
    result = {
        "time_range": np.array(res["time_range"]).round(6).tolist(),
        "detuning_range": np.array(res["detuning_range"]).round(6).tolist(),
        "frequencies": {k: float(v) for k, v in res["frequencies"].items()},
        "chevron_data": {
            k: np.array(v).round(6).tolist()
            for k, v in res["chevron_data"].items()
        },
        "rabi_rates": {
            k: np.array(v).round(6).tolist()
            for k, v in res["rabi_rates"].items()
        },
        "resonant_frequencies": {
            k: float(v) for k, v in res["resonant_frequencies"].items()
        },
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
