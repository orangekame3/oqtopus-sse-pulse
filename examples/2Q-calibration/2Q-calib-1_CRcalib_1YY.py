import traceback

from qubex.experiment import Experiment
from qubex.pulse import Pulse, PulseSchedule, FlatTop, VirtualZ
import numpy as np
import json

# 使用するqubitの設定
chip_id='64Qv3'
muxes=[9]

ctrl_qubit = 'Q36'
ctrl_qubit_frequency = 7.995820  # <-- ここを適切なqubit共鳴周波数に変更してください
ctrl_hpi_amplitude = 0.0273  # <-- ここを適切なhpi振幅に変更してください

trgt_qubit = 'Q37'
trgt_qubit_frequency = 8.975332  # <-- ここを適切なqubit共鳴周波数に変更してください
trgt_hpi_amplitude = 0.04  # <-- ここを適切なhpi振幅に変更してください

CR_ch = 'Q36-Q37'
CR_frequency = trgt_qubit_frequency

targets = [ctrl_qubit, trgt_qubit, CR_ch]  # 測定対象qubitリスト


# CR較正用パラメータ
CR_amplitude = 1. # CRパルス振幅は基本的にはフルパワー固定で問題ありません
CR_phase = 0.0  # <-- 較正結果を踏まえて更新してください
cncl_amplitude = 0.0  # <-- 較正結果を踏まえて更新してください
cncl_phase = 0.0  # <-- 較正結果を踏まえて更新してください

ctrl_state = '1'  # control qubitの初期状態 ('0' or '1')
ctrl_meas_basis = 'Y'  # control qubitの測定基底 ('X', 'Y', 'Z')
trgt_meas_basis = 'Y'  # target qubitの測定基底 ('X', 'Y', 'Z')

echo = False # echoシーケンスを使用する場合はTrueに設定


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
    with exp.modified_frequencies(
        {ctrl_qubit: ctrl_qubit_frequency,
         trgt_qubit: trgt_qubit_frequency,
         CR_ch: CR_frequency,}
        ):

        duration_list = np.arange(24, 1000, 20)  # CRパルス持続時間掃引リスト (2ns刻み)

        def CR_pulse(duration):
            return FlatTop(
                    duration = duration, # 2[ns]の倍数のみ
                    amplitude = CR_amplitude,
                    tau = 12,
                    ).shifted(CR_phase)
        
        def cncl_pulse(duration):
            return FlatTop(
                    duration = duration, # 2[ns]の倍数のみ
                    amplitude = cncl_amplitude,
                    tau = 12,
                    ).shifted(cncl_phase)
        
        ctrl_hpi_pulse = FlatTop(
                    duration = 32,
                    amplitude = ctrl_hpi_amplitude,
                    tau = 12,
                )
        
        trgt_hpi_pulse = FlatTop(
                    duration = 32,
                    amplitude = trgt_hpi_amplitude,
                    tau = 12,
                )

        # PulseScheduleクラスのhpi_repeatインスタンスを作成. 
        # 1つ引数が必要な関数のオブジェクト. 
        def CR_calib(duration: float) -> PulseSchedule:
            with PulseSchedule(targets) as ps:

                if ctrl_state == '1':
                    ps.add(ctrl_qubit, ctrl_hpi_pulse.repeated(2))  # control qubitを|1>に準備
                    ps.barrier()

                ps.add(CR_ch, CR_pulse(duration))
                ps.add(trgt_qubit, cncl_pulse(duration))
                ps.barrier() # バリアを入れて時間的に区切る

                if echo:
                    ps.add(ctrl_qubit, ctrl_hpi_pulse)
                    ps.barrier()

                    ps.add(CR_ch, CR_pulse(duration).shifted(np.pi))
                    ps.add(trgt_qubit, cncl_pulse(duration).shifted(np.pi))
                    ps.barrier()

                    ps.add(ctrl_qubit, ctrl_hpi_pulse)
                    ps.barrier()

                # X, Y測定のためのhpiパルスを追加
                if ctrl_meas_basis == 'X':
                    ps.add(ctrl_qubit, ctrl_hpi_pulse.shifted(np.pi/2))
                elif ctrl_meas_basis == 'Y':
                    ps.add(ctrl_qubit, ctrl_hpi_pulse)
                if trgt_meas_basis == 'X':
                    ps.add(trgt_qubit, trgt_hpi_pulse.shifted(np.pi/2))
                elif trgt_meas_basis == 'Y':
                    ps.add(trgt_qubit, trgt_hpi_pulse)

            return ps

        # 掃引が必要な実験では、sweep_parameterメソッドを使用するのが便利.
        res = exp.sweep_parameter(
            sequence = CR_calib, # 引数に関数を指定
            sweep_range = duration_list, # 掃引振幅リスト
        )


    # 結果を整形してJSON形式で出力
    result = {
        "duration_list": res.data[ctrl_qubit].sweep_range.tolist(),
        "ctrl_data_real": res.data[ctrl_qubit].data.real.tolist(),
        "ctrl_data_imag": res.data[ctrl_qubit].data.imag.tolist(),
        "trgt_data_real": res.data[trgt_qubit].data.real.tolist(),
        "trgt_data_imag": res.data[trgt_qubit].data.imag.tolist(),
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
