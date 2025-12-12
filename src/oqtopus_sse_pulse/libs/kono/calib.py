import qubex as qx
import json


def calibrate(ex: qx.Experiment):
    # calibrate
    ex.obtain_rabi_params(plot=False)
    ex.calibrate_hpi_pulse(plot=False)
    t1 = ex.t1_experiment(plot=False).data
    t2 = ex.t2_experiment(plot=False).data
    ex.build_classifier(plot=False)

    # put results into payload
    calib_note = ex.calib_note
    calib_note_dict = calib_note._dict if calib_note else None
    result: dict = {"calib_note": calib_note_dict}
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    props = {
        "t1": {
            key: t1[key].t1 for key in t1
        }, 
        "t1_err": {
            key: t1[key].t1_err for key in t1
        },
        "t1_r2": {
            key: t1[key].r2 for key in t1
        },
        "t2": {
            key: t2[key].t2 for key in t2
        },
        "t2_err": {
            key: t2[key].t2_err for key in t2
        },
        "t2_r2": {
            key: t2[key].r2 for key in t2
        }
    }
    result: dict = {"props": props}
    print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))