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
        params_dir="/sse/in/repo/ogawa/params",
        calib_note_path="/sse/in/repo/ogawa/calib_note.json"
    )

    # デバイスに接続
    exp.connect()

    # 空の波形リストを作成
    waveform = []

    # waveformリストを, qubexのPulseクラスのインスタンスに変換
    waveform = Pulse(waveform)

    # 波形シーケンスの辞書を作成
    sequence = {qubit: waveform}

    ampl_list = np.linspace(0.01, 0.25, 20)  # 読み出し振幅掃引リスト(0.25が最大)
    snr_list = []
    signal_list = []
    noise_list = []

    for ampl in ampl_list:
        # measureメソッドで測定を実行
        res = exp.measure(
            sequence = sequence, # 自作の波形シーケンスを指定
            mode = "single", # 単発射影測定の場合は"single"を指定
            shots = 1024, # ショット数
            readout_amplitudes = {qubit: ampl}, # 読み出し振幅を指定
        ) # MeasureResultクラスを出力する
        
        kerneled_data_real_list = res.data[qubit].raw.real
        kerneled_data_imag_list = res.data[qubit].raw.imag

        signal = np.mean(np.sqrt(np.array(kerneled_data_real_list)**2 + np.array(kerneled_data_imag_list)**2))
        noise = np.std(np.sqrt(np.array(kerneled_data_real_list)**2 + np.array(kerneled_data_imag_list)**2))
        snr = signal / noise
        snr = json.loads(json.dumps(snr, default=float))  # 強制Python化
        signal = json.loads(json.dumps(signal, default=float))  # 強制Python化
        noise = json.loads(json.dumps(noise, default=float))  # 強制Python化
        
        snr_list.append(snr)
        signal_list.append(signal)
        noise_list.append(noise)

    # 結果を整形してJSON形式で出力
    result = {
        "qubit": qubit,
        "ampl_list": ampl_list.tolist(),
        "snr_list": snr_list,
        "signal_list": signal_list,
        "noise_list": noise_list,
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
