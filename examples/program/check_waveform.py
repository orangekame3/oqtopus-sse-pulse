import traceback

from qubex.experiment import Experiment

print("start program")
try:
    username = "your_name"
    exp = Experiment(
        chip_id="64Qv2",
        muxes=[2],
        params_dir=f"/sse/in/repo/{username}/params"
    )
    exp.connect()
    res = exp.check_waveform(targets="Q08")
    result: dict = {"mode": "", "data": {}}
    for qubit, data in res.data.items():
        result["mode"] = res.mode.value
        result["data"][qubit] = {
            "raw": {
                "I": data.raw.real.tolist(), # type: ignore
                "Q": data.raw.imag.tolist(), # type: ignore
            },
            "kerneled": {
                "I": data.kerneled.real.tolist(), # type: ignore
                "Q": data.kerneled.imag.tolist(), # type: ignore
            },
        }
    print(f"payload={result}")
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()
finally:
    print("end program")
