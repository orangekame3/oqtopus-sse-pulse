import qubex as qx
import json


def calibrate(ex: qx.Experiment):
    # calibrate
    print(ex.system_manager._config_loader._params_dict)
    ex.obtain_rabi_params(plot=False)
    print(ex.system_manager._config_loader._params_dict)
    # ex.calibrate_hpi_pulse(plot=False)
    # t1 = ex.t1_experiment(plot=False)
    # t1 = t1.data
    # t2 = ex.t2_experiment(plot=False)
    # t2 = t2.data
    # cls = ex.build_classifier(plot=False)

    # # put results into payload
    # calib_note = ex.calib_note
    # calib_note_dict = calib_note._dict if calib_note else None
    # result: dict = {"calib_note": calib_note_dict}
    # print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    # props = {
    #     "t1": {
    #         key: t1[key].t1 for key in t1
    #     }, 
    #     "t1_err": {
    #         key: t1[key].t1_err for key in t1
    #     },
    #     "t1_r2": {
    #         key: t1[key].r2 for key in t1
    #     },
    #     "t2": {
    #         key: t2[key].t2 for key in t2
    #     },
    #     "t2_err": {
    #         key: t2[key].t2_err for key in t2
    #     },
    #     "t2_r2": {
    #         key: t2[key].r2 for key in t2
    #     },
    #     "readout_fidelities": cls["readout_fidelities"],
    #     "average_readout_fidelity": cls["average_readout_fidelity"],
    # }
    # result: dict = {"props": props}
    # print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    # misc: dict = {
    #     "data": cls["data"],
    #     "classifiers": cls["classifiers"]
    # }
    # result: dict = {"misc": misc}
    # print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    # return cls["classifier"]