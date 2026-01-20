import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, Blank
import numpy as np
import json

# 使用するqubitの設定
chip_id='64Qv3'
muxes=[9]
qubits = ['Q36','Q37','Q38','Q39','Q40','Q41','Q42','Q43']
# qubits = ['Q36','Q37','Q38','Q39']
# qubits = ['Q36','Q37','Q38','Q39','Q40','Q42']
#waits_ns_list =[600, 1200, 40, 60]
waits_ns_list =[600, 1200, 1800, 2400]
wait_ns = 600

print("start program")
try:
    # qubexのExperimentクラスのインスタンスを作成
    exp = Experiment(
        chip_id=chip_id,
#        muxes=muxes,
        qubits=qubits,
        params_dir="/sse/in/repo/hirose/params",
        calib_note_path="/sse/in/repo/hirose/calib_note.json"
    )

    # デバイスに接続
    exp.connect()

    # ラビ振動測定
    targets = qubits  # 測定対象qubitリスト
    time_range = np.arange(0, 200, 4) # 掃引時間リスト (単位: ns, 2nsより細かくはできない)
    ampl = 0.05 # パルス振幅(0~1の範囲の無次元相対量)
    readout_ampl = 0.25 # 読み出し振幅(max 0.25)

    # PulseScheduleクラスのrabi_sequenceインスタンスを作成. 
    # 1つ引数が必要な関数のオブジェクト. 
    def rabi_sequence(T: int) -> PulseSchedule:
        with PulseSchedule(targets) as ps:
            for target in targets:
                # ps.add(target, Blank(waits_ns_list[targets.index(target)%4]))
                ps.add(target, Pulse([ampl] * int(T/2)))  # 長さT nsの矩形波パルス (2nsサンプリングなので2で割っている)
    			# ps.add(target, Blank(wait_ns*(targets.index(target)%4)))  # 指定した待ち時間だけ待機
                # ps.add(target, Blank(waits_ns_list[targets.index(target)%4]))
        return ps

    # qubit毎のresonator周波数掃引範囲を設定
    base_frequencies = {
        'Q36': 10.200,
        'Q37': 10.200,
        'Q38': 10.200,
        'Q39': 10.200,
        'Q40': 10.200,
        'Q41': 10.200,
        'Q42': 10.200,
        'Q43': 10.200,
    }
    # base_frequencies = {
    #     'Q36': 10.200,
    #     'Q37': 10.200,
    #     'Q38': 10.200,
    #     'Q39': 10.200,
    #     'Q40': 10.200,
    #     'Q41': 10.200,
    #     'Q42': 10.200,
    #     'Q43': 10.200,
    # }

#    resonator_frequency_list = 10.1975 + np.linspace(-0.1, 0.1, 31)  # 読み出し周波数掃引リスト (GHz)
#    resonator_frequency_list = 10.1975 + np.linspace(-0.1, 0.1, 31)  # 読み出し周波数掃引リスト (GHz)
    # freq_sweep_range = np.linspace(-0.1, 0.1, 7)  # 各qubitの基準周波数からの掃引範囲 (GHz)
    freq_sweep_range = np.linspace(-0.1, 0.1, 31)  # 各qubitの基準周波数からの掃引範囲 (GHz)
    result_dict = {"qubits": {}}

    for qubit in qubits:
        result_dict["qubits"][qubit] = {
            "base_frequency": base_frequencies[qubit],
            "sweep_offset": freq_sweep_range.tolist(),
            "measurements": []
        }


    # 各qubitごとに同じoffsetで測定
    for offset in freq_sweep_range:
        # 各qubitの周波数を設定
        frequencies = {f'R{qubit}': base_frequencies[qubit] + offset for qubit in qubits}
        print(f"Measuring at offset: {offset}")

    # # 各qubitごとに異なるoffsetで測定
    # for i, offset in enumerate(freq_sweep_range):
    #     # 各qubitに異なるoffsetを割り当て (インデックスベース)
    #     frequencies = {f'R{qubit}': base_frequencies[qubit] + freq_sweep_range[(targets.index(qubit) + i) % len(freq_sweep_range)] for qubit in qubits}
    #    
    #     print(f"Measuring at offset index: {i}, offsets: {[freq_sweep_range[(targets.index(qubit) + i) % len(freq_sweep_range)] for qubit in qubits]}")

        # 掃引が必要な実験では、sweep_parameterメソッドを使用するのが便利.
        res = exp.sweep_parameter(
            sequence = rabi_sequence, # 引数に関数を指定
            sweep_range = time_range, # 掃引時間リスト
            frequencies=frequencies,
            readout_amplitudes = {qubit: readout_ampl for qubit in qubits},
            plot = False, # 測定中にプロット表示するかどうか
        )

        # 結果を整形
        for qubit in qubits:
            measurement_data = {
                "frequency": base_frequencies[qubit] + offset,
                "time_range": res.data[qubit].sweep_range.tolist(),
                "data_real": res.data[qubit].data.real.tolist(),
                "data_imag": res.data[qubit].data.imag.tolist(),
            }
            result_dict["qubits"][qubit]["measurements"].append(measurement_data)

    # 結果の出力
    print("payload=" + json.dumps(result_dict, ensure_ascii=False, separators=(",", ":")))

# 例外処理
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()

# 終了処理
finally:
    print("end program")
