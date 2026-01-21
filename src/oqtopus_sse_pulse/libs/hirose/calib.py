import qubex as qx
import json

from qubex.experiment.mixin.characterization_mixin import CharacterizationMixin
from qubex.experiment.mixin.calibration_mixin import CalibrationMixin
from qubex.experiment.experiment_result import ExperimentResult, AmplCalibData, T1Data, T2Data
from qubex.pulse import FlatTop, CPMG
from qubex.analysis import fitting
from qubex.pulse.pulse_schedule import PulseSchedule
from qubex.pulse.blank import Blank
from qubex.pulse.waveform import Waveform
from qubex.analysis import visualization as viz
from qubex.measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.experiment_constants import CALIBRATION_SHOTS, HPI_DURATION, HPI_RAMPTIME, PI_DURATION, PI_RAMPTIME
from qubex.backend.device_controller import SAMPLING_PERIOD
from typing import Collection
import numpy as np
from numpy.typing import ArrayLike

import traceback

import math

from typing import Literal


class CustomCharacterizationMixin(CharacterizationMixin):
    # define the custom version of calibrate_control_frequency()
    def calibrate_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict[str, float]:
        result = self.chevron_pattern(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            frequencies=frequencies,
            amplitudes=amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=False,   # do not save image
        )
        resonant_frequencies = result["resonant_frequencies"]

        print("\nResults\n-------")
        print("ge frequency (GHz):")
        for target, frequency in resonant_frequencies.items():
            print(f"    {target}: {frequency:.6f}")
        return resonant_frequencies
    

    def t1_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
        readout_amplitudes: dict[str, float] | None = None,
    ) -> ExperimentResult[T1Data]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        self.validate_rabi_params(targets)

        if time_range is None:
            time_range = np.logspace(
                np.log10(100),
                np.log10(100 * 1000),
                51,
            )
        time_range = self.util.discretize_time_range(np.asarray(time_range))

        data: dict[str, T1Data] = {}

        subgroups = self.util.create_qubit_subgroups(targets)
        print(f"Target qubits: {targets}")
        print(f"Subgroups: {subgroups}")
        for idx, subgroup in enumerate(subgroups):
            if len(subgroup) == 0:
                continue

            def t1_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(subgroup) as ps:
                    for target in subgroup:
                        ps.add(target, self.get_hpi_pulse(target).repeated(2))
                        ps.add(target, Blank(T))
                return ps

            print(
                f"({idx + 1}/{len(subgroups)}) Conducting T1 experiment for {subgroup}...\n"
            )

            sweep_result = self.sweep_parameter(
                sequence=t1_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
                title="T1 decay",
                xlabel="Time (μs)",
                ylabel="Measured value",
                xaxis_type=xaxis_type,
                readout_amplitudes=readout_amplitudes,  # enable custom readout amplitudes
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="T1",
                    xlabel="Time (μs)",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                )
                if fit_result["status"] == "success":
                    t1 = fit_result["tau"]
                    t1_err = fit_result["tau_err"]
                    r2 = fit_result["r2"]

                    t1_data = T1Data.new(
                        sweep_data,
                        t1=t1,
                        t1_err=t1_err,
                        r2=r2,
                    )
                    data[target] = t1_data

                    fig = fit_result["fig"]

                    if save_image:
                        viz.save_figure_image(
                            fig,
                            name=f"t1_{target}",
                        )

        return ExperimentResult(data=data)

    def t2_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        n_cpmg: int = 1,
        pi_cpmg: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        readout_amplitudes: dict[str, float] | None = None,
    ) -> ExperimentResult[T2Data]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        self.validate_rabi_params(targets)

        if time_range is None:
            time_range = np.logspace(
                np.log10(300),
                np.log10(100 * 1000),
                51,
            )
        time_range = self.util.discretize_time_range(
            time_range=np.asarray(time_range),
            sampling_period=2 * SAMPLING_PERIOD,
        )

        data: dict[str, T2Data] = {}

        subgroups = self.util.create_qubit_subgroups(targets)

        print(f"Target qubits: {targets}")
        print(f"Subgroups: {subgroups}")
        for idx, subgroup in enumerate(subgroups):
            if len(subgroup) == 0:
                continue

            def t2_sequence(T: int) -> PulseSchedule:
                with PulseSchedule(subgroup) as ps:
                    for target in subgroup:
                        hpi = self.get_hpi_pulse(target)
                        pi = pi_cpmg or hpi.repeated(2)
                        ps.add(target, hpi)
                        if T > 0:
                            ps.add(
                                target,
                                CPMG(
                                    tau=(T - pi.duration * n_cpmg) // (2 * n_cpmg),
                                    pi=pi,
                                    n=n_cpmg,
                                ),
                            )
                        ps.add(target, hpi.shifted(np.pi))
                return ps

            print(
                f"({idx + 1}/{len(subgroups)}) Conducting T2 experiment for {subgroup}...\n"
            )

            sweep_result = self.sweep_parameter(
                sequence=t2_sequence,
                sweep_range=time_range,
                shots=shots,
                interval=interval,
                plot=plot,
                readout_amplitudes=readout_amplitudes,  # enable custom readout amplitudes
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_exp_decay(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=0.5 * (1 - sweep_data.normalized),
                    plot=plot,
                    title="T2 echo",
                    xlabel="Time (μs)",
                    ylabel="Normalized signal",
                )
                if fit_result["status"] == "success":
                    t2 = fit_result["tau"]
                    t2_err = fit_result["tau_err"]
                    r2 = fit_result["r2"]

                    t2_data = T2Data.new(
                        sweep_data,
                        t2=t2,
                        t2_err=t2_err,
                        r2=r2,
                    )
                    data[target] = t2_data

                    fig = fit_result["fig"]

                    if save_image:
                        viz.save_figure_image(
                            fig,
                            name=f"t2_echo_{target}",
                        )

        return ExperimentResult(data=data)
    

class CustomCalibrationMixin(CalibrationMixin):
    def calibrate_default_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        update_params: bool = True,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        readout_amplitudes: dict[str, float] | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=duration if duration is not None else HPI_DURATION,
                    amplitude=1,
                    tau=ramptime if ramptime is not None else HPI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=duration if duration is not None else PI_DURATION,
                    amplitude=1,
                    tau=ramptime if ramptime is not None else PI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")

            # calculate the control amplitude for the target rabi rate
            ampl = self.calc_control_amplitude(target, rabi_rate)

            # create a range of amplitudes around the estimated value
            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = np.clip(ampl_min, 0, 1)
            ampl_max = np.clip(ampl_max, 0, 1)
            if ampl_min == ampl_max:
                ampl_min = 0
                ampl_max = 1
            ampl_range = np.linspace(ampl_min, ampl_max, n_points)

            n_per_rotation = 2 if pulse_type == "pi" else 4

            sweep_data = self.sweep_parameter(
                sequence=lambda x: {target: pulse.scaled(x)},
                sweep_range=ampl_range,
                readout_amplitudes=readout_amplitudes,  # enable custom readout amplitudes
                repetitions=n_per_rotation * n_rotations,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            fit_result = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=sweep_data.normalized,
                plot=plot,
                title=f"{pulse_type} pulse calibration",
                ylabel="Normalized signal",
            )

            r2 = fit_result["r2"]
            if r2 > r2_threshold:
                if update_params:
                    if pulse_type == "hpi":
                        self.calib_note.update_hpi_param(
                            target,
                            {
                                "target": target,
                                "duration": pulse.duration,
                                "amplitude": fit_result["amplitude"],
                                "tau": pulse.tau,
                            },
                        )
                    elif pulse_type == "pi":
                        self.calib_note.update_pi_param(
                            target,
                            {
                                "target": target,
                                "duration": pulse.duration,
                                "amplitude": fit_result["amplitude"],
                                "tau": pulse.tau,
                            },
                        )
            else:
                print(f"Error: R² value is too low ({r2:.3f})")
                print(f"Calibration data not stored for {target}.")

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=fit_result["amplitude"],
                r2=r2,
            )

        data: dict[str, AmplCalibData] = {}
        for target in targets:
            if target not in self.calib_note.rabi_params:
                print(f"Rabi parameters are not stored for {target}.")
                continue
            print(f"Calibrating {pulse_type} pulse for {target}...")
            data[target] = calibrate(target)

        print("")
        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"  {target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)


    def calibrate_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        readout_amplitudes: dict[str, float] | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        return self.calibrate_default_pulse(
            targets=targets,
            pulse_type="hpi",
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,  # enable custom readout amplitudes
        )

    def calibrate_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        readout_amplitudes: dict[str, float] | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        return self.calibrate_default_pulse(
            targets=targets,
            pulse_type="pi",
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,  # enable custom readout amplitudes
        )

class CustomExperiment(CustomCharacterizationMixin, CustomCalibrationMixin, qx.Experiment):
# class CustomExperiment(CustomCharacterizationMixin, qx.Experiment):
    # inherit the custom calibrate_control_frequency function by extending CustomCharacterizationMixin
    pass


def calibrate(ex: CustomExperiment):
    try:
        # start calibration
        ex.obtain_rabi_params(plot=False)                                                               # Rabi measurement

        # detect errors in Rabi measurement
        err_qubits = []
        for qubit, rabi_params in ex.calib_note._dict["rabi_params"].items():
            if rabi_params["frequency"] is None or math.isnan(rabi_params["frequency"]) or np.isnan(rabi_params["frequency"]):
                err_qubits.append(qubit)
        if len(err_qubits) > 0:
            # stop calibration and output error message if Rabi measurement failed for some qubits
            result: dict = {"status": "failed", "err_qubits": err_qubits}
            # result: dict = {"status": "error", "err_qubits": err_qubits}
            print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            raise RuntimeError(f"Rabi measurement failed for qubits: {', '.join(err_qubits)}")

        # continue calibration if Rabi measurement terminated successfully
        control_frequencies = ex.calibrate_control_frequency(plot=False)                                # calibrate qubit frequencies
        ex.modified_frequencies(control_frequencies)                                                    # update qubit frequencies
        # control_amplitude = {}
        # for qubit in ex.qubit_labels:
        #     qres = ex.measure_qubit_resonance(target=qubit, plot=False, save_image=False)             # measure qubit resonance
        #     control_amplitude[qubit] = qres["estimated_amplitude"]

        readout_amplitude = {}
        for qubit in ex.qubit_labels:
            res = ex.find_optimal_readout_amplitude(target=qubit, plot=False, save_image=False) # obtain readout amplitudes
            readout_amplitude[qubit] = res["optimal_amplitude"]

        print("Warning! just measures readout frequencies, not runs calibration")
        readout_frequencies = ex.calibrate_readout_frequency(targets=ex.qubit_labels, readout_amplitudes=readout_amplitude)           # obtain readout frequencies
        # readout_frequencies = ex.calibrate_readout_frequency(targets=ex.qubit_labels)                   # calibrate readout frequencies

        ex.calibrate_hpi_pulse(plot=False, readout_amplitudes=readout_amplitude)                                                      # calibrate hpi pulse
        ex.calibrate_pi_pulse(plot=False, readout_amplitudes=readout_amplitude)                                                      # calibrate pi pulse
        t1 = ex.t1_experiment(plot=False, readout_amplitudes=readout_amplitude)                                                       # T1 measurement
        t1 = t1.data                                                                            # store results of T1 measurement
        t2 = ex.t2_experiment(plot=False, readout_amplitudes=readout_amplitude)                                                       # T2 measurement
        t2 = t2.data                                                                            # store results of T2 measurement
        cls = ex.build_classifier(plot=False, readout_amplitudes=readout_amplitude)                                                   # build classifiers
        # ex.calibrate_hpi_pulse(plot=False)                                                              # calibrate hpi pulse
        # t1 = ex.t1_experiment(plot=False)                                                               # T1 measurement
        # t1 = t1.data                                                                                    # store results of T1 measurement
        # t2 = ex.t2_experiment(plot=False)                                                               # T2 measurement
        # t2 = t2.data                                                                                    # store results of T2 measurement
        # cls = ex.build_classifier(plot=False)                                                           # build classifiers

        # summarize results
        calib_note = ex.calib_note
        calib_note_dict = calib_note._dict if calib_note else None

        props = {
            "resonator_frequency": {
                key: readout_frequencies[key] for key in readout_frequencies
            },
            "qubit_frequency": control_frequencies,
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
            },
            "readout_fidelities": cls["readout_fidelities"],
            "average_readout_fidelity": cls["average_readout_fidelity"],
        }

        params = {
            key: readout_amplitude[key] for key in readout_amplitude
        }

        # output
        result: dict = {"status": "succeeded", "calib_note": calib_note_dict, "props": props, "params": params}
        # result: dict = {"status": "succeeded", "calib_note": calib_note_dict, "props": props}
        print("payload=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))

    # handle exceptions
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()
