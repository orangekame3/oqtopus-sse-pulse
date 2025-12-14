import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, FlatTop, Blank
import numpy as np
import json

# 使用するqubitの設定
chip_id='64Qv3'
muxes=[9,10]
qubit = ['Q36']
#qubit = ['Q36', 'Q37']
#qubit = ['Q36','Q37','Q38','Q39','Q40']
time_idle = 500  # 各hpiパルス間の待ち時間(ns)
counts_mes = 100  # 測定回数

print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
        muxes=muxes,
        params_dir="/sse/in/repo/hirose/params", # <-- 自分のparamsディレクトリのパスに変更してください
        calib_note_path="/sse/in/repo/hirose/calib_note.json" # <-- 自分のcalib_noteファイルのパスに変更してください
    )

    # デバイスに接続
    exp.connect()

    calib_note = exp.calib_note
    # Convert CalibrationNote to dict for JSON serialization
    calib_note_dict = calib_note._dict if calib_note else None

    targets = qubit  # 測定対象qubitリスト
#    time_list = np.linspace(0, 20_000, 51)  # 待ち時間掃引リスト (例: 0から20µsまでを51ステップで掃引)
#    time_list = np.full(50, 2000)  # すべて2000ns固定
#    time_list = np.full(50, 1000)  # すべて1000ns固定

#    time_idle = 1000  # 各hpiパルス間の待ち時間(ns)
#    counts_mes = 20  # 測定回数
    
#    time_list = np.full(counts_mes, time_idle)  # すべて1000ns固定

    # 各qubitに対応するhpiパルスを作成
    hpi_pulse = {}
    pi_pulse = {}
    for target in targets:
        hpi_pulse[target] = FlatTop(
                                    duration=32,
                                    amplitude=calib_note_dict['hpi_params'][target]['amplitude'],
                                    tau=12,
                                )
        pi_pulse[target] = FlatTop(
                                    duration=32,
                                    amplitude=calib_note_dict['pi_params'][target]['amplitude'],
                                    tau=12,
                                )


    # waveformリストを, qubexのPulseクラスのインスタンスに変換
    # waveform = Pulse(waveform)

    # 波形シーケンスの辞書を作成
    # sequence = {targets: pi_pulse}
    
    def sequence(time_idle: int) -> PulseSchedule:
        with PulseSchedule(targets) as ps:
            for target in targets:
#                ps.add(target, hpi_pulse[target])
#                ps.add(target, hpi_pulse[target])
                ps.add(target, pi_pulse[target])
                ps.add(target, Blank(time_idle))  # 指定した待ち時間だけ待機
        return ps

    # with PulseSchedule(targets) as ps:
    #     for target in targets:
    #         ps.add(target, hpi_pulse[target])
    #         ps.add(target, hpi_pulse[target])
    #         ps.add(target, Blank(time_idle))  # 指定した待ち時間だけ待機

    # PulseScheduleクラスのhpi_repeatインスタンスを作成. 
    # 1つ引数が必要な関数のオブジェクト. 
    # def ltmeas(time: float) -> PulseSchedule:
    #     with PulseSchedule(targets) as ps:
    #         for target in targets:
    #             ps.add(target, hpi_pulse[target])
    #             ps.add(target, hpi_pulse[target])
    #             ps.add(target, Blank(time))  # 指定した待ち時間だけ待機
    #     return ps

    # # 掃引が必要な実験では、sweep_parameterメソッドを使用するのが便利.
    # res = exp.sweep_parameter(
    #     sequence = ltmeas, # 引数に関数を指定
    #     sweep_range = time_list, # 掃引リスト
    # )

    # measureメソッドで測定を実行
    res = exp.measure(
#        interval = 150*1000*2,
        sequence = sequence(time_idle), # 自作の波形シーケンスを指定
        mode = "single", # 単発射影測定の場合は"single"を指定
        shots = counts_mes # ショット数
    ) # MeasureResultクラスを出力する


    # 結果を整形してJSON形式で出力
    result = {}
#    result["time_list"] = (np.arange(len(res.data[target].raw)) * time_list).tolist(),  # 読み出しのサンプリング間隔は8ns
    for target in targets:
        result[target] = {
            # "time_list": res.data[target].sweep_range.tolist(),
            # "data_real": res.data[target].data.real.tolist(),
            # "data_imag": res.data[target].data.imag.tolist(),
#            "time_range": (np.arange(len(res.data[target].raw)) * 8).tolist(),  # 読み出しのサンプリング間隔は8ns
#            "raw_data_real": res.data[target].raw.real.tolist(),
#            "raw_data_imag": res.data[target].raw.imag.tolist(),
#            "time_range": (np.arange(len(res.data[target].raw)) * 8).tolist(),  # 読み出しのサンプリング間隔は8ns
#            "time_list": time_list.tolist(),
#            "time_list": (np.arange(len(res.data[target].raw)) * time_idle).tolist(),  # 読み出しのサンプリング間隔は8ns
            "time_list": (np.arange(len(res.data[target].raw)) * (time_idle if time_idle != 0 else 1)).tolist(),  # 読み出しのサンプリング間隔は8ns
#            "time_list": (np.arange(len(res.data[target].raw))).tolist(),  # 読み出しのサンプリング間隔は8ns
            # "data_real": (res.data[target].kerneled.real / len(res.data[target].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
            # "data_imag": (res.data[target].kerneled.imag / len(res.data[target].raw)).tolist(),  # kerneledデータは合計値なので, 平均値に変換
            "data_real": res.data[target].raw.real.tolist(),
            "data_imag": res.data[target].raw.imag.tolist(),
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
