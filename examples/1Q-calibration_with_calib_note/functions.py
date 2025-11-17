"""
last modified: 2024/06/30
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
from scipy.optimize import curve_fit
from IPython.display import clear_output
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def PCA_and_rotation(complex_list):
    """
    複素数のリストを入力として, PCA(主成分分析)による回転変換を行う.

    Parameters
    ----------
    complex_list : list
        複素数のリスト

    Returns
    -------
    rotated_complex_list : list
        回転後の複素数のリスト
    rotation_angle : float
        回転角度[rad]
    """
    # Convert the imput list data to a 2D vector 
    data = np.column_stack([np.real(complex_list), np.imag(complex_list)])

    # Perform PCA
    pca = PCA(n_components=1)
    pca.fit(data)

    # Get the principal components
    components = pca.components_
    explained_variance = pca.explained_variance_

    # Calculate the slope and intercept of the line
    slope = components[0, 1] / components[0, 0]
    intercept = np.mean(data[:, 1]) - slope * np.mean(data[:, 0])

    # Rotate the original data to align with the y-axis
    rotation_angle = np.arctan(slope) + np.pi / 2
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotated_data = np.dot(data, rotation_matrix)

    # Convert the rotated data to a list
    rotated_real_list = rotated_data[:, 0].tolist()
    rotated_imag_list = rotated_data[:, 1].tolist()

    rotated_complex_list = np.array(rotated_real_list) + 1j * np.array(rotated_imag_list)

    return rotated_complex_list, rotation_angle




def fourier_transform(
        x_list: np.ndarray,
        y_list: np.ndarray,
        ) -> tuple:
    """
    Perform the Fourier transform of the input data.

    Parameters
    ----------
    x_list : np.ndarray
        The x-axis data.
    y_list : np.ndarray
        The y-axis data.

    Returns
    -------
    x_fourier_list : np.ndarray
        The x-axis data of the Fourier transform.
    y_fourier_list : np.ndarray
        The y-axis data of the Fourier transform.
    """
    
    Nf = 2**16
    y_list_ = y_list - np.average(y_list) # オフセットを除去 
    y_list = np.pad(y_list_, (0, Nf - len(y_list))) # y_listの長さがNfとなるように0でパディング

    # Perform the Fourier transform
    y_fourier_list = np.fft.fft(y_list)
    x_fourier_list = np.fft.fftfreq(Nf, d=x_list[1] - x_list[0])

    # y_fourier_listが左右対称のため、左半分のみを取得
    y_fourier_list = y_fourier_list[:len(y_fourier_list) // 2]
    x_fourier_list = x_fourier_list[:len(x_fourier_list) // 2]

    return x_fourier_list, y_fourier_list


def find_frequency(
    x_list: np.ndarray,
    y_list: np.ndarray,
    ) -> float:
    """
    Find the frequency of the input signal.

    Parameters
    ----------
    x_list : np.ndarray
        The x-axis data.
    y_list : np.ndarray
        The y-axis data.

    Returns
    -------
    frequency : float
        The frequency of the input signal.
    """
    x_fourier_list, y_fourier_list = fourier_transform(x_list, y_list)
    frequency = x_fourier_list[np.argmax(np.abs(y_fourier_list))]

    return frequency


def damped_oscillation(t, A, omega, tau, offset, phi):
    """
    減衰振動関数.

    Parameters
    ----------
    t : float
        時間[ns]
    A : float
        振幅
    omega : float
        Rabi周波数[GHz]
    tau : float
        減衰時間[ns]
    offset : float
        オフセット
    phi : float
        位相オフセット[rad]
        
    Returns
    -------
    float
        減衰振動関数の値
    """
    return A * np.exp(-t/tau) * np.cos(omega*t + phi) + offset


def normalized_damped_oscillation(t, omega, tau, phi):
    """
    規格化された減衰振動関数.

    Parameters
    ----------
    t : float
        時間[ns]
    omega : float
        Rabi周波数[GHz]
    tau : float
        減衰時間[ns]
    phi : float
        位相オフセット[rad]
        
    Returns
    -------
    float
        減衰振動関数の値
    """
    return np.exp(-t/tau) * np.cos(omega*t + phi)


def fit_damped_oscillation(
        t_list, 
        data_list,
        freq_guess: float = 0.0125,):
    """
    測定結果を減衰振動関数でフィッティングする.

    Parameters
    ----------
    t_list : list
        時間のリスト
    data_list : list
        測定結果のリスト
    freq_guess : float
        Rabi周波数の初期値[GHz]
    
    Returns
    -------
    A_fit : float
        振幅
    omega_fit : float
        Rabi周波数[GHz]
    tau_fit : float
        減衰時間[ns]
    offset_fit : float
        オフセット
    phi_fit : float
        位相オフセット[rad]
    """

    # Define the initial parameters
    p0=np.arange(5, dtype=float)
    if data_list[0] > data_list[1]:
        sign = 1
    else:
        sign = -1
    p0[0] = sign * (np.max(data_list) - np.min(data_list)) # amplitude
    p0[1] = freq_guess * 2 * np.pi # Rabi frequency [GHz]
    p0[2] = 1000 # decay time [ns]
    p0[3] = np.average(data_list) # offset
    p0[4] = 0 # phase offset [rad]

    # Fit the data to the damped oscillation function
    popt, pcov = curve_fit(damped_oscillation, t_list, data_list, p0=p0, maxfev=100000)

    # Extract the fitting parameters
    A_fit, omega_fit, tau_fit, offset_fit, phi_fit = popt

    return A_fit, omega_fit, tau_fit, offset_fit, phi_fit


def fit_normalized_damped_oscillation(
        t_list, 
        normalized_data_list,
        freq_guess: float = 0.0125,):
    """
    測定結果を[+1,-1]で規格化された減衰振動関数でフィッティングする.

    Parameters
    ----------
    t_list : list
        時間のリスト
    normalized_data_list : list
        測定結果のリスト
    freq_guess : float
        Rabi周波数の初期値[GHz]
    
    Returns
    -------
    omega_fit : float
        Rabi周波数[GHz]
    tau_fit : float
        減衰時間[ns]
    phi_fit : float
        位相オフセット[rad]
    """

    # Define the initial parameters
    p0=np.arange(3, dtype=float)
    p0[0] = freq_guess * 2 * np.pi # Rabi frequency [GHz]
    p0[1] = 1000 # decay time [ns]
    p0[2] = 0 # phase offset [rad]

    # Fit the data to the damped oscillation function
    popt, pcov = curve_fit(normalized_damped_oscillation, t_list, normalized_data_list, p0=p0, maxfev=100000)

    # Extract the fitting parameters
    omega_fit, tau_fit, phi_fit = popt

    return omega_fit, tau_fit, phi_fit


def Rabi_normalization(data_list, A_fit, offset_fit):
    """
    測定結果をフィッティング結果で規格化する.

    Parameters
    ----------
    data_list : list
        測定結果のリスト
    A_fit : float
        振幅
    offset_fit : float
        オフセット
    
    Returns
    -------
    normalized_data : list
        フィッティング結果で規格化された測定結果のリスト
    """

    # Normalize the oscillation to [-1, 1]
    normalized_data = (data_list - offset_fit) / A_fit
    
    return normalized_data


# リストの要素間をcons倍に補完する関数
def list_filling(sparse_list, cons):
    new_diff = (sparse_list[1] - sparse_list[0]) / cons
    dense_list = np.arange(sparse_list[0], sparse_list[-1], new_diff)
    return dense_list



def get_local_maxima_idx(sig_list):
    # 最初の極大値のインデックスを取得
    for i in range(len(sig_list)-3):
        diffs = [sig_list[i] - sig_list[i+j] for j in range(1, 5)]
        if all([diff > 0 for diff in diffs]):
            break
    return i


def get_local_minima_idx(sig_list):
    # 最初の極小値のインデックスを取得
    for i in range(len(sig_list)-3):
        diffs = [sig_list[i] - sig_list[i+j] for j in range(1, 5)]
        if all([diff < 0 for diff in diffs]):
            break
    return i


def get_fit_params_for_rabi(
    time_range: np.ndarray,
    rotated_results: dict,
    graph: bool = True,
    ) -> dict:
    """
    測定結果を減衰振動関数でフィッティングする.
    
    Parameters
    ----------
    time_range: np.ndarray
        Rabi実験で変化させるパラメータの値の配列
    rotated_results: dict
        測定結果の辞書
    graph: bool
        グラフを表示するかどうか
    
    Returns
    -------
    A_fit: dict
        振幅の辞書
    offset_fit: dict
        オフセットの辞書
    rabi_freq_fit: dict
        Rabi周波数の辞書
    """

    fit_params: dict = {key: [] for key in rotated_results.keys()}

    dense_time_range = list_filling(sparse_list=time_range, cons=10)

    for key in rotated_results.keys():

        freq_guess = find_frequency(time_range, rotated_results[key])

        fit_params[key] = fit_damped_oscillation(time_range, rotated_results[key], freq_guess)
        normalized_data = Rabi_normalization(rotated_results[key], fit_params[key][0], fit_params[key][3])
        normalized_fit = Rabi_normalization(
            damped_oscillation(dense_time_range, *fit_params[key]), 
            fit_params[key][0], 
            fit_params[key][3]
            )

        if graph:
            plt.figure(figsize=(6, 4))
            plt.scatter(time_range, normalized_data, label="normalized data")
            plt.plot(dense_time_range, normalized_fit, label="fit")
            plt.legend()
            plt.xlabel("duration / ns")
            plt.ylabel("Z expectaiton value")
            plt.title(f"{key}")
            plt.grid()
            plt.show()

    A_fit: dict = {key: fit_params[key][0] for key in rotated_results.keys()}
    offset_fit: dict = {key: fit_params[key][3] for key in rotated_results.keys()}
    rabi_freq_fit: dict = {key: fit_params[key][1] for key in rotated_results.keys()}

    return A_fit, offset_fit, rabi_freq_fit


def rotation_conversion_with_angle(complex_list, rotation_angle):
    """
    実部虚部のリストを入力として, 与えられた角度での回転変換を行う.
    
    Parameters
    ----------
    complex_list : list
        複素数のリスト
    rotation_angle : float
        回転角度[rad]

    Returns
    -------
    rotated_complex_list : list
        回転後の複素数のリスト
    """

    # Convert the imput list data to a 2D vector 
    data = np.column_stack([np.real(complex_list), np.imag(complex_list)])

    # Rotate the original data to align with the y-axis
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotated_data = np.dot(data, rotation_matrix)

    # Convert the rotated data to a list
    rotated_real_list = np.array(rotated_data[:, 0].tolist())
    rotated_imag_list = np.array(rotated_data[:, 1].tolist())

    return rotated_real_list + 1j * rotated_imag_list


def get_fit_params_for_pi(
    ampl_range: np.ndarray,
    signals: dict,
    graph: bool = True,
    ) -> dict:
    """
    測定結果を減衰振動関数でフィッティングする.
    
    Parameters
    ----------
    ampl_range: np.ndarray
        駆動振幅値の配列
    signals: dict
        測定結果の辞書
    graph: bool
        グラフを表示するかどうか

    Returns
    -------
    ampl_for_pi: dict
        πパルスとなる振幅値の辞書
    """

    fit_params: dict = {key: [] for key in signals.keys()}

    dense_ampl_range = list_filling(sparse_list=ampl_range, cons=10)

    for key in signals.keys():

        # local_maxima_idx = get_local_maxima_idx(signals[key])
        # local_minima_idx = get_local_minima_idx(signals[key])
        # expected_period = np.abs(ampl_range[local_minima_idx] - ampl_range[local_maxima_idx]) * 2
        freq_guess = find_frequency(ampl_range, signals[key])

        fit_params[key] = fit_damped_oscillation(ampl_range, signals[key], freq_guess)
        normalized_data = Rabi_normalization(signals[key], fit_params[key][0], fit_params[key][3])
        normalized_fit = Rabi_normalization(
            damped_oscillation(dense_ampl_range, *fit_params[key]), 
            fit_params[key][0], 
            fit_params[key][3]
            )

        if graph:
            plt.figure(figsize=(6, 4))
            plt.scatter(ampl_range, normalized_data, label="normalized data")
            plt.plot(dense_ampl_range, normalized_fit, label="fit")
            plt.legend()
            plt.xlabel("Amplitude")
            plt.ylabel("Z expectaiton value")
            plt.title(f"{key}")
            plt.grid()
            plt.show()

    ampl_for_pi: dict = {key[1:4]: np.pi / fit_params[key][1] for key in signals.keys()}

    return ampl_for_pi


def linear_fitting(x, y):
    """
    線形回帰を行う.

    Parameters
    ----------
    x : np.ndarray
        xの配列
    y : np.ndarray
        yの配列

    Returns
    -------
    y_fit : np.ndarray
        予測値
    grad : float
        傾き
    intercept : float
        y切片
    """
    model = LinearRegression() # 線形回帰モデル
    x_T = x[:,np.newaxis] # 縦ベクトルに変換する必要あり
    model.fit(x_T, y) # モデルを訓練データに適合, 引数は縦ベクトルでないといけない
    y_fit = model.predict(x_T) # 引数は縦ベクトルでないといけない
    grad = model.coef_ # 傾き
    intercept = model.intercept_ # y切片
    return y_fit, grad, intercept


def get_slope_and_intercept_by_PCA(complex_list):
        """
        複素数のリストからPCAによって直線の傾きと切片を取得する関数.
        
        Parameters
        ----------
        complex_list : list
            複素数のリスト.

        Returns
        -------
        slope : float
            直線の傾き.
        intercept : float
            直線の切片.
        """

        # Convert the imput list data to a 2D vector 
        data = np.column_stack([np.real(complex_list), np.imag(complex_list)])

        # Perform PCA
        pca = PCA(n_components=1)
        pca.fit(data)

        # Get the principal components
        components = pca.components_
        explained_variance = pca.explained_variance_

        # Calculate the slope and intercept of the line
        slope = components[0, 1] / components[0, 0]
        intercept = np.mean(data[:, 1]) - slope * np.mean(data[:, 0])

        return slope, intercept



def plot_ge_iq_distribution(
        qubit,
        result_ge_list_dict, 
        graph=True,
        ):
    """
    単発射影測定における測定結果をプロットする関数.

    Parameters
    ----------
    qubit : str
        測定対象のqubit.
    result_ge_list_dict : list
        測定結果のリストの辞書.
        {'g': [...], 'e': [...]}という形式
    graph : bool
        グラフを表示するかどうか. デフォルトはTrue.

    Returns
    -------
    rotation_angle : float
        虚軸に揃えるための回転角.
    """
    if graph:
        plt.figure(figsize=(5, 5))

    width_list = []
    ave_point_list = []
    for e, state in enumerate(['g', 'e']):
        if graph:
            plt.scatter(
                np.real(result_ge_list_dict[state]), 
                np.imag(result_ge_list_dict[state]), 
                label=f'{state}, measurement', 
                s=1,
                # alpha=0.2, 
                )
        width = np.max(np.abs(result_ge_list_dict[state]))
        width_list.append(width)

        ave_point = np.mean(result_ge_list_dict[state])
        ave_point_list.append(ave_point)
        if graph:
            plt.scatter(
                ave_point.real, 
                ave_point.imag, 
                label=f'{state}, average', 
                s=50,
                color='black',
                marker='*', 
                )

    width = np.max(width_list)
    if graph:
        plt.xlim(-width, width)
        plt.ylim(-width, width)
        plt.plot(np.linspace(-width, width, 2), np.zeros(2), linewidth = 1, color='black')
        plt.plot(np.zeros(2), np.linspace(-width, width, 2), linewidth = 1, color='black')

    # 回転角の取得
    slope, intercept = get_slope_and_intercept_by_PCA(ave_point_list)
    rotation_angle = np.arctan(slope) + np.pi / 2
    
    x_list = np.linspace(-1e9, 1e9, 100)
    if graph:
        plt.plot(
            x_list, 
            slope * x_list + intercept, 
            label='projection line', 
            color='black', 
            linestyle='dotted'
            )

        plt.legend()
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title(f'{qubit}')
        plt.grid()

        plt.tight_layout()
        plt.show()

    return rotation_angle



def plot_histogram_for_projection_measurement(
        qubit,
        rotated_result_ge_list_dict,
        semilogy=False,
        graph=True,
        ):
    """
    単発射影測定における測定結果をヒストグラムでプロットする関数.

    Parameters
    ----------
    qubit : str
        測定対象のqubit.
    rotated_result_ge_list_dict : dict
        測定結果のリストの辞書.
    semilogy : bool
        y軸を対数スケールにするかどうか. デフォルトはFalse.
    graph : bool
        グラフを表示するかどうか. デフォルトはTrue.

    Returns
    -------
    ther_ex_ratio : float
        熱励起率.
    backaction_ratio : float
        反作用率.
    x_th : float
        閾値.
    ge_reverse : bool
        g, eの分布が逆転しているかどうか.
    fidelity : float
        フィデリティ.
    """

        
    # ヒストグラムの作成
    if graph:
        plt.figure(figsize=(6, 4))
    width_g = max(np.abs(rotated_result_ge_list_dict['g']))
    width_e = max(np.abs(rotated_result_ge_list_dict['e']))
    width = max(width_g, width_e)
    bin_list = np.arange(np.min([rotated_result_ge_list_dict['g'], rotated_result_ge_list_dict['e']]), 
                         np.max([rotated_result_ge_list_dict['g'], rotated_result_ge_list_dict['e']]), 
                         width/30)

    hist_x_list = []
    hist_y_list = []
    peak_position_list = []

    for e, state in enumerate(['g', 'e']):
        hist = np.histogram(rotated_result_ge_list_dict[state], bins=bin_list)
        hist_y, hist_x = hist
        hist_x_list.append(hist_x)
        hist_y = np.append(hist_y, 0) # y軸は要素数が1少ないので要素“0"を加える
        hist_y_list.append(hist_y)
        if graph:
            plt.bar(hist_x, hist_y, alpha = 0.4, width=width/30, label=f'{state}, measurement')

        # 分布のガウシアンフィッティング
        # 2つ山ガウシアンの定義
        def func(x, p_large, p_small, mu_large, mu_small, sigma):
            return p_large * np.exp(-(x-mu_large)**2 / (2*sigma**2)) + p_small * np.exp(-(x-mu_small)**2 / (2*sigma**2))
        
        # フィッティングの初期値
        shot_num = len(rotated_result_ge_list_dict[state])
    
        for i, x in enumerate(hist_x):
            if hist_y[i] > np.max(hist_y)/2:
                x_left = x
                break
        x_center = hist_x[np.argmax(hist_y)]
        sigma_ini = np.abs(x_center - x_left) # ガウシアンの標準偏差

        p_large_ini = shot_num/10 # 大きいガウシアンの振幅
        p_small_ini = shot_num/100 # 小さいガウシアンの振幅
        mu_large_ini = np.mean(rotated_result_ge_list_dict[state]) # 大きいガウシアンの平均値
        mu_small_ini = np.mean(rotated_result_ge_list_dict['e'] if state=='g' else rotated_result_ge_list_dict['g']) # 小さいガウシアンの平均値
        param_ini = np.array([p_large_ini, p_small_ini, mu_large_ini, mu_small_ini, sigma_ini])
        # print(f'param_ini = {param_ini}')

        popt, pcov = curve_fit(func, hist_x, hist_y, p0 = param_ini, maxfev = 100000, )
        # print(f'popt = {popt}')
        hist_fit = func(hist_x, *popt)
        if graph:
            plt.plot(hist_x, hist_fit, label=f'{state}, fitting')

        # 熱励起率の算出
        if state == 'g':
            p_large, p_small, mu_large, mu_small, sigma = popt
            ther_ex_ratio = p_small / (p_large + p_small)

        # 反作用率の算出
        if state == 'e':
            p_large, p_small, mu_large, mu_small, sigma = popt
            backaction_ratio = p_small / (p_large + p_small) - ther_ex_ratio

        peak_position_list.append(mu_large)

    # thresholdの算出
    x_th = np.average(peak_position_list)
    if graph:
        plt.vlines(x_th, 0, np.max(hist_y_list), color='green', label='threshold')

    # フィデリティの算出
    sum_minus = np.sum(hist_y_list[0][hist_x_list[0] < x_th]) 
    sum_plus = np.sum(hist_y_list[1][hist_x_list[1] > x_th])
    fidelity = (2*shot_num - (sum_minus + sum_plus)) / (2*shot_num)
    ge_reverse = False
    if fidelity < 0.5:
        fidelity = 1 - fidelity
        ge_reverse = True

    if graph:
        if semilogy:
            plt.semilogy() 
            plt.ylim(0.5, 2e3)
        
        plt.xlabel('Displacement')
        plt.ylabel('Counts')
        plt.title(f'{qubit}')
        plt.legend()
        plt.grid()
        plt.show()

    return ther_ex_ratio, backaction_ratio, x_th, ge_reverse, fidelity


def plot_histogram_for_projection_measurement_calibration(
        qubit,
        rotated_result_ge_list_dict,
        semilogy=False,
        graph=True,
        ):
    """
    単発射影測定における測定結果をヒストグラムでプロットする関数.

    Parameters
    ----------
    qubit : str
        測定対象のqubit.
    rotated_result_ge_list_dict : dict
        測定結果のリストの辞書.
    semilogy : bool
        y軸を対数スケールにするかどうか. デフォルトはFalse.
    graph : bool
        グラフを表示するかどうか. デフォルトはTrue.

    Returns
    -------
    ther_ex_ratio : float
        熱励起率.
    backaction_ratio : float
        反作用率.
    x_th : float
        閾値.
    ge_reverse : bool
        g, eの分布が逆転しているかどうか.
    fidelity : float
        フィデリティ.
    """

        
    # ヒストグラムの作成
    if graph:
        plt.figure(figsize=(6, 4))
    width_g = max(np.abs(rotated_result_ge_list_dict['g']))
    width_e = max(np.abs(rotated_result_ge_list_dict['e']))
    width = max(width_g, width_e)
    bin_list = np.arange(np.min([rotated_result_ge_list_dict['g'], rotated_result_ge_list_dict['e']]), 
                         np.max([rotated_result_ge_list_dict['g'], rotated_result_ge_list_dict['e']]), 
                         width/30)

    hist_x_list = []
    hist_y_list = []
    peak_position_list = []

    for e, state in enumerate(['g', 'e']):
        hist = np.histogram(rotated_result_ge_list_dict[state], bins=bin_list)
        hist_y, hist_x = hist
        hist_x_list.append(hist_x)
        hist_y = np.append(hist_y, 0) # y軸は要素数が1少ないので要素“0"を加える
        hist_y_list.append(hist_y)
        if graph:
            plt.bar(hist_x, hist_y, alpha = 0.4, width=width/30, label=f'{state}, measurement')

    # thresholdの算出
    for xi in hist_x_list[0]:
        sum_g = np.sum(hist_y_list[0][hist_x_list[0] < xi]) 
        sum_e = np.sum(hist_y_list[1][hist_x_list[1] > xi])
        if sum_g > sum_e:
            x_th = xi
            break

    if graph:
        plt.vlines(x_th, 0, np.max(hist_y_list), color='green', label='threshold')

    # フィデリティの算出
    shot_num = len(rotated_result_ge_list_dict['g'])
    sum_minus = np.sum(hist_y_list[0][hist_x_list[0] < x_th]) 
    sum_plus = np.sum(hist_y_list[1][hist_x_list[1] > x_th])
    fidelity = (2*shot_num - (sum_minus + sum_plus)) / (2*shot_num)
    ge_reverse = False
    if fidelity < 0.5:
        fidelity = 1 - fidelity
        ge_reverse = True

    if graph:
        if semilogy:
            plt.semilogy() 
            plt.ylim(0.5, 2e3)
        
        plt.xlabel('Displacement')
        plt.ylabel('Counts')
        plt.title(f'{qubit}')
        plt.legend()
        plt.grid()
        plt.show()

    return x_th, ge_reverse, fidelity


def outcome_discrimination(
        iq_value: float,
        rotation_angle: float,
        threshold: float,
        ge_reverse: bool,
        ) -> str:
    """
    単発読み出しの結果を0/1判定する関数.

    Parameters
    ----------
    iq_value : float
        読み出しで取得したIQ値.
    rotation_angle : float
        g,eの分布を虚軸方向に揃えるための回転角.
    threshold : float
        g,eを識別するための閾値.
    ge_reverse : bool
        g, eの分布が逆転しているかどうか.

    Returns
    -------
    result : str
        測定結果. '0' or '1'.
    """
    rotated_value = np.imag(rotation_conversion_with_angle([iq_value], rotation_angle)[0])
       
    if ge_reverse:
        outcome_0 = '1'
        outcome_1 = '0'
    else:
        outcome_0 = '0'
        outcome_1 = '1'

    if rotated_value > threshold:
        reuslt = outcome_0
    else:
        reuslt = outcome_1

    return reuslt


def get_single_proj_meas(
    single_iq_list: list,
    qubit: str,
    rotation_angle_dict: dict,
    threshold_dict: dict,
    ge_reverse_dict: dict,
    ) -> list:
    """
    単発射影測定の結果を取得する関数.

    Parameters
    ----------
    single_iq_list : list
        単発読み出しによるIQ値の, shot回数分の要素が並んだリスト.
    qubit : str
        ターゲットのqubit名.
    rotation_angle_dict : dict
        g,eの分布を虚軸方向に揃えるための回転角の辞書.
    threshold_dict : dict
        g,eを識別するための閾値の辞書.
    ge_reverse_dict : dict
        g, eの分布が逆転しているかどうかの辞書.

    Returns
    -------
    result_list : list
        測定結果のリスト. ['0', '0', '1', '0', ...]のような形.
    """

    shot_num = len(single_iq_list)
    result_list = []

    for i in range(shot_num):
        result_list.append(outcome_discrimination(
            single_iq_list[i],
            rotation_angle_dict[qubit],
            threshold_dict[qubit],
            ge_reverse_dict[qubit],
            ))
    
    return result_list


from itertools import product
def state_combination_list(qubit_num):
    """
    ['00...0', '10...0', '11...0', ...]のような, 2^qubit_num個の状態のリストを作成する.

    Parameters
    ----------
    qubit_num : int
        qubitの数

    Returns
    -------
    combinations : list
        2^qubit_num個の状態のリスト
    """
    state_list = ['0', '1']
    combinations = [''.join(comb) for comb in product(state_list, repeat=qubit_num)]

    return combinations


def search_spec_qubit(qubit_name_str, one_side_size):
    """
    上下左右のspecutator qubitのqubit番号を出力する関数. 
    qubit_name_strは'Q00'のような, 'Q'+数2桁という形式を仮定する. 
    one_side_sizeは使用するqubitチップの1辺のqubit数.
    """
    Q = int(qubit_name_str[1:3]) # qubit番号の取り出し
    MUX = Q // 4 # MUX番号の取得
    q = Q % 4 # MUX中4つのqubitのうちのどれなのかの取得
    m = int(one_side_size / 2) # 1辺のMUX数

    if not (Q < one_side_size**2 and Q >= 0):
        raise Exception('Input qubit number is incorrect.')

    if q==0:
        Q_up    = 4*(MUX-1*m) + 2
        Q_left  = 4*(MUX-1  ) + 1
        Q_right = 4*(MUX+0  ) + 1
        Q_down  = 4*(MUX+0*m) + 2

    elif q==1:
        Q_up    = 4*(MUX-1*m) + 3
        Q_left  = 4*(MUX-0  ) + 0
        Q_right = 4*(MUX+1  ) + 0
        Q_down  = 4*(MUX+0*m) + 3

    elif q==2:
        Q_up    = 4*(MUX-0*m) + 0
        Q_left  = 4*(MUX-1  ) + 3
        Q_right = 4*(MUX+0  ) + 3
        Q_down  = 4*(MUX+1*m) + 0

    else:
        Q_up    = 4*(MUX-0*m) + 1
        Q_left  = 4*(MUX-0  ) + 2
        Q_right = 4*(MUX+1  ) + 2
        Q_down  = 4*(MUX+1*m) + 1

    # 端のqubitの処理
    MUX_top_list       = [i for i in range(m)] # 一番上のMUX番号リスト
    MUX_left_end_list  = [i*m for i in range(m)] # 一番左のMUX番号リスト
    MUX_right_end_list = [i*m + m-1 for i in range(m)] # 一番右のMUX番号リスト
    MUX_bottom_list    = [(m-1)*m + i for i in range(m)] # 一番下のMUX番号リスト

    if (MUX in MUX_top_list) and (q==0 or q==1):
        Q_up = np.nan
    if (MUX in MUX_left_end_list) and (q==0 or q==2):
        Q_left = np.nan
    if (MUX in MUX_right_end_list) and (q==1 or q==3):
        Q_right = np.nan
    if (MUX in MUX_bottom_list) and (q==2 or q==3):
        Q_down = np.nan

    # int型のqubit番号を'Q00'のようなstr型に変換する
    Q_list = [Q_up, Q_left, Q_right, Q_down]
    Q_name_list = []
    
    for Q_ in Q_list:
        if not np.isnan(Q_):
            Q_name_list.append(f'Q{Q_:02}')

    return Q_name_list


def freq_calc_ctrl(qubit_name, freq_info, mode, one_side_size, acceptable_freq_range):
    """
    qubit名, freq_infoを入力すると, 適切なLO, NCOの値を出力してくれる関数.
    LOは8000〜13000MHzの範囲で, 500MHz単位で設定しないといけない.
    NCOは1500〜3000MHzの範囲で, 375/16 MHz = 23.4375 MHz単位で設定しないといけない.  
    
    Parameters
    ----------
    qubit_name : str
        qubit名. 'Q00'のような形式を仮定する.
    freq_info : dict
        qubitの周波数情報をまとめた辞書. ただしGHz単位.
    mode : str
        LSB/USBのどちらかを指定する.
    one_side_size : int
        使用するqubitチップの1辺のqubit数.
    acceptable_freq_range : list
        許容できる周波数範囲([最低値, 最高値]のリスト形式. ただしGHz単位).

    Returns
    -------
    [f_lo, f_cnco, f_fnco0, f_fnco1, f_fnco2]というリスト (MHz単位).
    これらはlo.mhz, nco.mhz, awg0.nco.mhz, awg1.nco.mhz, awg2.nco.mhzに代入する値に対応する.
    awg0でge周波数, awg1,2を跨いだ範囲(400MHz × 2)でCR周波数をカバーできるように割り当てる. 
    CRのチャンネルの割り当ては, CR_group_dictとして以下の形式で出力される:
    {
        'Q03-01': 'CQ03_1',
    }
    これを用いると, 後でlogic_ch[CR_group_dict['Q03-01']]などと記入すれば適切な論理チャンネルが選ばれる. 
    """
    
    # LSB/USBの切り替え
    if mode=='LSB':
        pm = -1
    elif mode=='USB':
        pm = 1 

    f_ef = freq_info[qubit_name]['ef'] * 1e3 # ef周波数(この関数では使わない)
    f_ge = freq_info[qubit_name]['ge'] * 1e3 # ge周波数
    CR_freq_list = freq_info[qubit_name]['CR'].copy() # CR周波数リスト(GHz単位)
    CR_freq_list = [CR_freq * 1e3 for CR_freq in CR_freq_list] # MHz単位に変換

    # acceptable_freq_rangeの範囲外の周波数の処理
    if f_ge < acceptable_freq_range[0] * 1e3:
        f_ge = acceptable_freq_range[0] * 1e3
    elif f_ge > acceptable_freq_range[1] * 1e3:
        f_ge = acceptable_freq_range[1] * 1e3
    for i, CR_freq in enumerate(CR_freq_list):
        if CR_freq < acceptable_freq_range[0] * 1e3:
            CR_freq_list[i] = acceptable_freq_range[0] * 1e3
        elif CR_freq > acceptable_freq_range[1] * 1e3: 
            CR_freq_list[i] = acceptable_freq_range[1] * 1e3
            
    # nan入力の処理
    if np.isnan(f_ef):
        f_ef = 10000 + pm * 2250 - 400 # エラーが出ない無難な値
    if np.isnan(f_ge):
        f_ge = 10000 + pm * 2250 # エラーが出ない無難な値
    for f, CR_freq in enumerate(CR_freq_list):
        if np.isnan(CR_freq):
            CR_freq_list[f] = 10000 + pm * 2250 # エラーが出ない無難な値

    
    # CR周波数のグルーピング
    sorted_CR_freq_list = np.sort(CR_freq_list) #昇順に並び替え

    position_list = [] # 「昇順に並び替えたとき, 各要素は元のリストの何番目の要素か」を表すリスト
    for freq in sorted_CR_freq_list:
        for i, freq2 in enumerate(CR_freq_list):
            if freq==freq2:
                position_list.append(i)

    if len(CR_freq_list)==4:
        D01 = np.abs(sorted_CR_freq_list[0] - sorted_CR_freq_list[1]) # 各周波数間の離調
        D02 = np.abs(sorted_CR_freq_list[0] - sorted_CR_freq_list[2])
        D03 = np.abs(sorted_CR_freq_list[0] - sorted_CR_freq_list[3])
        D12 = np.abs(sorted_CR_freq_list[1] - sorted_CR_freq_list[2])
        D13 = np.abs(sorted_CR_freq_list[1] - sorted_CR_freq_list[3])
        D23 = np.abs(sorted_CR_freq_list[2] - sorted_CR_freq_list[3])
    
        if D01 > D13:
            CR_lo_num_list = [position_list[0]] # 元のCR_freq_listの要素番号でのグルーピング結果
            CR_hi_num_list = [position_list[1], position_list[2], position_list[3]]
        elif D02 < D23:
            CR_lo_num_list = [position_list[0], position_list[1], position_list[2]]
            CR_hi_num_list = [position_list[3]]
        else:
            CR_lo_num_list = [position_list[0], position_list[1]]
            CR_hi_num_list = [position_list[2], position_list[3]]

    elif len(CR_freq_list)==3:
        D01 = np.abs(sorted_CR_freq_list[0] - sorted_CR_freq_list[1]) # 各周波数間の離調
        D02 = np.abs(sorted_CR_freq_list[0] - sorted_CR_freq_list[2])
        D12 = np.abs(sorted_CR_freq_list[1] - sorted_CR_freq_list[2])
    
        if D01 > D12:
            CR_lo_num_list = [position_list[0]] # 元のCR_freq_listの要素番号でのグルーピング結果
            CR_hi_num_list = [position_list[1], position_list[2]]
        else:
            CR_lo_num_list = [position_list[0], position_list[1]]
            CR_hi_num_list = [position_list[2]]

    elif len(CR_freq_list)==2:
        CR_lo_num_list = [position_list[0]]
        CR_hi_num_list = [position_list[1]]


    CR_freq_lo_list = [CR_freq_list[i] for i in CR_lo_num_list]
    CR_freq_hi_list = [CR_freq_list[i] for i in CR_hi_num_list]

    CR_group_dict = {}
    for i, spec_qubit in enumerate(search_spec_qubit(qubit_name, one_side_size)):
        if i in CR_lo_num_list:
            CR_group_dict[qubit_name+'-'+spec_qubit] = f'C{qubit_name}_1'
        elif i in CR_hi_num_list:
            CR_group_dict[qubit_name+'-'+spec_qubit] = f'C{qubit_name}_2'
    
    
    # 各周波数の決定        
    f_cnco = 23.4375 * 96 # 固定値(2250)
    f_c = (np.min([f_ge]+CR_freq_list) + np.max([f_ge]+CR_freq_list)) / 2 # 全体の中央の周波数
    f_lo_nco = 8000 + pm * f_cnco # f_lo +/- f_cncoの周波数 (探索のための初期値)


    # 500MHz単位でf_cに最も近いf_lo_ncoを探す. 
    while True:
        f_diff = np.abs(f_c - f_lo_nco)
        if f_diff <= 250:
            break
        f_lo_nco += 500
        if f_lo_nco > 13000 + pm * f_cnco:
            raise Exception(f'No suitable f_lo_nco found. qubit={qubit_name}, f_ge={f_ge}, CR_freq_list={CR_freq_list}') # LOは8000〜13000MHzを超えるとエラー. 

    f_lo = f_lo_nco - pm * f_cnco

    # 23.4375MHz単位で, 適切なf_fnco0を探す. 
    f_fnco0 = -23.4375 * 42 # (=-987.1875)
    while True:
        f_diff = np.abs(f_lo + pm * (f_cnco + f_fnco0) - f_ge)
        if f_diff <= 23.4375/2:
            break
        f_fnco0 += 23.4375
        if f_fnco0 > 1000:
            raise Exception(f'No suitable f_fnco0 found. qubit={qubit_name}, f_ge={f_ge}, CR_freq_list={CR_freq_list}') # NCOは1500〜3000MHzを超えるとエラー. 

    f1 = (np.min(CR_freq_lo_list) + np.max(CR_freq_lo_list)) / 2
    f_fnco1 = -23.4375 * 42 # (=-987.1875)
    while True:
        f_diff = np.abs(f_lo + pm * (f_cnco + f_fnco1) - f1)
        if f_diff <= 23.4375/2:
            break
        f_fnco1 += 23.4375
        if f_fnco1 > 1000:
            raise Exception(f'No suitable f_fnco1 found. qubit={qubit_name}, f_ge={f_ge}, CR_freq_list={CR_freq_list}')

    f2 = (np.min(CR_freq_hi_list) + np.max(CR_freq_hi_list)) / 2
    f_fnco2 = -23.4375 * 42 # (=-987.1875)
    while True:
        f_diff = np.abs(f_lo + pm * (f_cnco + f_fnco2) - f2)
        if f_diff <= 23.4375/2:
            break
        f_fnco2 += 23.4375
        if f_fnco2 > 1000:
            raise Exception(f'No suitable f_fnco2 found. qubit={qubit_name}, f_ge={f_ge}, CR_freq_list={CR_freq_list}')
    
    return [f_lo, f_cnco, f_fnco0, f_fnco1, f_fnco2], CR_group_dict



def freq_calc_ro(mux_name, freq_info, mode):
    """
    qubit名, freq_infoを入力すると, 適切なLO, NCOの値を出力してくれる関数.
    LOは8000〜13000MHzの範囲で, 500MHz単位で設定しないといけない.
    NCOは1500〜3000MHzの範囲で, 375/16 MHz = 23.4375 MHz単位で設定しないといけない.  
    
    Parameters
    ----------
    mux_name : str
        MUX名. 'MUX00'のような形式.
    freq_info : dict
        freq_infoの辞書. ただしGHz単位.
    mode : str
        LSB/USBの切り替え. 'LSB'または'USB'.

    Returns
    -------
    [f_lo, f_cnco, f_fnco0]というリスト(MHz単位).
    これらはlo.mhz, nco.mhz, awg0.nco.mhzに代入する値に対応する.
    """
    
    # LSB/USBの切り替え
    if mode=='LSB':
        pm = -1
    elif mode=='USB':
        pm = 1 

    ro_freq_list = freq_info[mux_name].copy() # 読み出し周波数リスト(GHz単位)
    ro_freq_list = [ro_freq * 1e3 for ro_freq in ro_freq_list] # MHz単位に変換
        
    # nan入力の処理
    for i, ro_freq in enumerate(ro_freq_list):
        if np.isnan(ro_freq):
            ro_freq_list[i] = 10100 # エラーが出ない無難な値
    
    
    # 各周波数の決定        
    f_cnco = 23.4375 * 85 # 固定値(1992.1875)
    f_c = (np.min(ro_freq_list) + np.max(ro_freq_list)) / 2 # 全体の中央の周波数
    f_lo_nco = 8000 + pm * f_cnco # f_lo +/- f_cncoの周波数 (探索のための初期値)

    # print(ro_freq_list)
    
    # 500MHz単位でf_cに最も近いf_lo_ncoを探す. 
    while True:
        f_diff = np.abs(f_c - f_lo_nco)
        if f_diff <= 250:
            break
        f_lo_nco += 500
        if f_lo_nco > 13000 + pm * f_cnco:
            raise Exception(f'No suitable f_lo_nco found. mux={mux_name}, f_c={f_c}') # LOは8000〜13000MHzを超えるとエラー. 

    f_lo = f_lo_nco - pm * f_cnco

    # 23.4375MHz単位で, 適切なf_fnco0を探す. 
    f_fnco0 = -23.4375*32 # (=-750)
    while True:
        f_diff = np.abs(f_lo + pm * (f_cnco + f_fnco0) - f_c)
        if f_diff <= 23.4375/2:
            break
        f_fnco0 += 23.4375
        if f_fnco0 > 750:
            raise Exception(f'No suitable f_fnco0 found. mux={mux_name}, f_c={f_c}') # NCOは1500〜3000MHzを超えるとエラー. 
    
    return [f_lo, f_cnco, f_fnco0]


def meas_raw_data_to_prob_dist(
        single_shot_capture_data: dict,
        single_meas_conditions: dict,
        ) -> list:
    """
    captureしたqubit数分のbit状態の出現確率分布リストを生成する関数.

    Parameters
    ----------
    single_shot_capture_data : dict
        単発射影測定の結果の辞書. 
    single_meas_conditions : dict
        単発射影測定の条件の辞書.

    Returns
    -------
    result_ : list
        bit状態の出現確率dictのリスト.
        [{'0000': 0.1, '0001': 0.2, ...},  <-- 1つ目のcapture結果
         {'0000': 0.1, '0001': 0.2, ...},  <-- 2つ目のcapture結果
          ...]
        のような形.
    """
    data = single_shot_capture_data
    readout_target_list = list(data.keys())
    rotation_angle_dict = single_meas_conditions['rotation_angle']
    threshold_dict = single_meas_conditions['projection_threshold']
    ge_reverse_dict = single_meas_conditions['ge_reverse']
    capture_num = len(data[readout_target_list[0]]) 
    shot_num = data[readout_target_list[0]][0].shape[1]

    # iq値を0/1に変換
    # {'Q00': [
    #           ['0', '1', '0', '1', ...], <-- 1つ目のcapture結果
    #           ['1', '0', '1', '0', ...], <-- 2つ目のcapture結果
    #           ...
    #         ], 
    # 'Q01': [
    #           ['0', '1', '0', '1', ...], <-- 1つ目のcapture結果
    #           ['1', '0', '1', '0', ...], <-- 2つ目のcapture結果
    #           ...
    #         ], 
    # ...}
    # という形の辞書を生成
    zero_one_dict = {}
    for readout_target, iqs in data.items():
        
        zero_one_list_ = []
        for cap in range(capture_num):

            single_iq_list = np.mean(iqs[cap], axis=0) # shot_num個のIQ値のリスト         
            zero_one_list = get_single_proj_meas(
                single_iq_list,
                R2C(readout_target),
                rotation_angle_dict,
                threshold_dict,
                ge_reverse_dict,
                )
            zero_one_list_.append(zero_one_list)
        
        zero_one_dict[readout_target] = zero_one_list_

    # captureごとの0/1リストをreadout_targetごとにまとめる
    # [
    #  {'Q00': ['0', '1', '0', '1', ...], 'Q01': ['0', '1', '0', '1', ...], ...}, <-- 1つ目のcapture結果
    #  {'Q00': ['0', '1', '0', '1', ...], 'Q01': ['0', '1', '0', '1', ...], ...}, <-- 2つ目のcapture結果
    #  ...
    # ]
    # という形のリストを生成
    zero_one_dict_list = []

    for cap in range(capture_num):
        zero_one_dict_list.append({}) # listの各要素に空の辞書を追加

        for readout_target, zero_one_list in zero_one_dict.items():
            zero_one_dict_list[cap].update({readout_target: zero_one_list[cap]})

    # captureごとの複数bit状態listを要素としたlistを生成
    # [
    #  ['0010', '1010', '0011', ...], <-- 1つ目のcapture結果
    #  ['1110', '0010', '0000', ...], <-- 2つ目のcapture結果
    #  ...
    # ]
    # という形のリストを生成
    zero_one_digit_list_list = []

    for zero_one_dict_ in zero_one_dict_list: 
        # zero_one_dict_ = {'Q00': ['0', '1', '0', '1', ...], 'Q01': ['0', '1', '0', '1', ...], ...}
        
        zero_one_digit_list = []
        for i in range(shot_num):
            
            zero_one_digit = ''
            for zero_one_list_2 in zero_one_dict_.values():
                # zero_one_list_2 = ['0', '1', '0', '1', ...]

                zero_one_digit += zero_one_list_2[i]

            zero_one_digit_list.append(zero_one_digit)

        zero_one_digit_list_list.append(zero_one_digit_list)

    # 各bit状態の出現確率リストを生成
    # [
    #  {'0000': 0.1, '0001': 0.2, ...}, <-- 1つ目のcapture結果
    #  {'0000': 0.1, '0001': 0.2, ...}, <-- 2つ目のcapture結果
    #  ...
    # ]
    # という形のリストを生成
    result_ = []
    for zero_one_digit_list_ in zero_one_digit_list_list:
        # zero_one_digit_list_ = ['0010', '1010', '0011',...]
    
        result = {}
        for zero_one_state in state_combination_list(len(readout_target_list)):
            result[zero_one_state] = zero_one_digit_list_.count(zero_one_state) / shot_num
        result_.append(result)

    return result_


def probs_to_multiple_Z_expectation(
        prob_dict: dict,
        meas_observable: str,
        ):
    """
    複数量子ビットの射影結果をZZ...の期待値に変換する関数.
    
    Parameters
    ----------
    prob_dict: dict
        射影結果を格納した辞書. {'00': 0.4, '01': 0.1, '10': 0.1, '11': 0.4}のような形式.
    meas_observable: str
        測定するオブザーバブル. 'Z'と'I'の組み合わせで作成する. 'ZZIZI'のような形式.
        
    Returns
    -------
    expectation_value: float
        ZZ...の期待値
    """  

    for state in prob_dict.keys():
        for i, ge in enumerate(state):
            if meas_observable[i]=='Z':
                if ge == '1':
                    prob_dict[state] *= -1

    # probsのvalueの総和
    expectation_value = sum(prob_dict.values())      

    return expectation_value


def confusion_matrix_inv(single_qubit_meas_prob_matrix_dict):
    """
    与えられた読み出しチャンネル分の混同行列の逆行列を計算する関数

    Parameters
    ----------
    single_qubit_meas_prob_matrix_dict: dict
        1qubit毎の混同行列の辞書
        
    Returns
    -------
    tensor_product: ndarray
        与えられた読み出しチャンネル分の混同行列の逆行列のテンソル積
    """
    
    tensor_product = [1] 

    for key in single_qubit_meas_prob_matrix_dict.keys():

        for i in range(2):
            single_qubit_meas_prob_matrix_dict[key][i] =  single_qubit_meas_prob_matrix_dict[key][i] / np.sum(single_qubit_meas_prob_matrix_dict[key][i])
            
        single_confusion_matrix_inv = np.linalg.inv(np.array(single_qubit_meas_prob_matrix_dict[key]).T)
        tensor_product = np.kron(tensor_product, single_confusion_matrix_inv)

    return tensor_product


from qutip import Bloch
from mpl_toolkits.mplot3d import Axes3D

def draw_bloch(XYZ_list):
    """
    Bloch球にリストデータを表示.

    Parameters
    ----------
    XYZ_list: list
        リストデータ
    """

    fig = plt.figure(figsize=(20, 10))
    view_list = [[0,0], [90,0], [0,90], [30,30]]

    points = [XYZ_list[0], XYZ_list[1], XYZ_list[2]]
    start = [XYZ_list[0][0], XYZ_list[1][0], XYZ_list[2][0]]

    for i, view in enumerate(view_list):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.view_init(view[0], view[1])

        b = Bloch(axes=ax)
        b.add_points(points)
        b.add_points(start)
        # b.point_color = ['r', 'm']
        b.point_size = [20, 200]
        b.make_sphere()


def draw_bloch_double(XYZ_list_g, XYZ_list_e):
    """
    Bloch球にリストデータを表示.

    Parameters
    ----------
    XYZ_list_g: list
        リストデータ
    XYZ_list_e: list
        リストデータ
    """

    fig = plt.figure(figsize=(20, 10))
    view_list = [[0,0], [90,0], [0,90], [30,30]]

    points_g = [XYZ_list_g[0], XYZ_list_g[1], XYZ_list_g[2]]
    points_e = [XYZ_list_e[0], XYZ_list_e[1], XYZ_list_e[2]]
    start_g = [XYZ_list_g[0][0], XYZ_list_g[1][0], XYZ_list_g[2][0]]
    start_e = [XYZ_list_e[0][0], XYZ_list_e[1][0], XYZ_list_e[2][0]]
    
    for i, view in enumerate(view_list):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.view_init(view[0], view[1])
        
        b = Bloch(axes=ax)
        b.add_points(points_g)
        b.add_points(points_e)
        b.add_points(start_g)
        b.add_points(start_e)
        # b.point_color = ['r', 'b', 'm', 'c']
        b.point_size = [20, 20, 200, 200]
        b.make_sphere()


def search_spec_qubit_in_give_set(
    central_qubit_list: list,
    qubit_list: list,
    one_side_size: int,
    ) -> list:
    """
    与えられたセット内の上下左右のspecutator qubitのqubit番号を出力する関数.
    """
    spec_qubit_list = []

    for central_qubit in central_qubit_list:    
        # qubit_listの中でcentral_qubitの上下左右にあるqubitを探す
        spec_qubit_list_new = [q_ for q_ in search_spec_qubit(central_qubit, one_side_size) if q_ in qubit_list]
        # 前のループにおけるspec_qubit_listとの重複を除去
        spec_qubit_list = list(set(spec_qubit_list) | set(spec_qubit_list_new))

    for central_qubit in central_qubit_list:
        # central_qubit自身があるなら除去
        if central_qubit in spec_qubit_list:
            spec_qubit_list.remove(central_qubit)
    
    return spec_qubit_list



def rabi_freq_to_ampl(
    qubit: str,
    rabi_frequency: float,
    ) -> float:
    """
    ラビ周波数からパルス振幅を計算する関数

    Parameters
    ----------
    qubit: str
        qubitの名前
    rabi_frequency: float
        ラビ周波数
    
    Returns
    -------
    amplitude: float
        パルス振幅
    """
    amplitude = rabi_frequency * ampl_per_rabi_freq_dict[qubit]

    return amplitude


def ampl_to_rabi_freq(
    qubit: str,
    amplitude: float,
    ) -> float:
    """
    パルス振幅からラビ周波数を計算する関数

    Parameters
    ----------
    qubit: str
        qubitの名前
    amplitude: float
        パルス振幅(0〜1)
    
    Returns
    -------
    rabi_frequency: float
        ラビ周波数 [Hz]
    """
    rabi_frequency  = amplitude / ampl_per_rabi_freq_dict[qubit]

    return rabi_frequency


def AC_Stark_shift(f_detuning, f_ampl):
    """
    Parameters
    ----------
    f_detuning : float
        離調 (駆動周波数-共鳴周波数)
    f_ampl : float
        駆動振幅

    Returns
    -------
    float
        AC Stark shift
    """
    return np.sqrt(f_detuning**2 + f_ampl**2) - f_detuning


def EPG_in_coherence_limit_1Q(
        t_gate,
        T1,
        T2_star
        ):
    """
    1QubitのEPGの計算.

    Parameters
    ----------
    t_gate : float
        ゲート時間
    T1 : float
        T1緩和時間
    T2_star : float
        Ramsey緩和時間

    Returns
    -------
    float
        EPG
    """
    EPG = (1/6) * (3 - np.exp(-t_gate/T1) - 2 * np.exp(-t_gate/T2_star))
    return EPG


def EPG_in_coherence_limit_2Q(
        t_gate,
        T1_Q0,
        T1_Q1,
        T2_star_Q0,
        T2_star_Q1,
        ):
    """
    2QubitのEPGの計算.

    Parameters
    ----------
    t_gate : float
        ゲート時間
    T1_Qi : float
        qubit i のT1緩和時間
    T2_star_Qi : float
        qubit i のRamsey緩和時間

    Returns
    -------
    float
        EPG
    """
    EPG = (1/20) * (15 - np.exp(-t_gate/T1_Q0) - 2 * np.exp(-t_gate/T2_star_Q0)
                    - np.exp(-t_gate/T1_Q1) - 2 * np.exp(-t_gate/T2_star_Q1)
                    - np.exp(-t_gate * (1/T1_Q0 + 1/T1_Q1)) - 4 * np.exp(-t_gate * (1/T2_star_Q0 + 1/T2_star_Q1))
                    - 2 * np.exp(-t_gate * (1/T1_Q0 + 1/T2_star_Q1)) - 2 * np.exp(-t_gate * (1/T2_star_Q0 + 1/T1_Q1))
                    )
    return EPG