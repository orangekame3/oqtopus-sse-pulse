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
    res = exp.check_rabi(targets="Q08")
    result: dict = {"mode": "", "data": {}}
    for qubit, data in res.data.items():
        result["data"][qubit] = {
            "raw": {
                "I": data.data.real.tolist(), # type: ignore
                "Q": data.data.imag.tolist(), # type: ignore
            },
            "normalized": {
                "I": data.normalized.real.tolist(), # type: ignore
                "Q": data.normalized.imag.tolist(), # type: ignore
            }
        }
    print(f"payload={result}")

except Exception as e:
    print("Exception:", e)
    traceback.print_exc()
finally:
    print("end program")
