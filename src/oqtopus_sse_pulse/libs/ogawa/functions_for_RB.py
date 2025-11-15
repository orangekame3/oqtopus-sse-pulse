import numpy as np
import random
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



# クリフォード群の生成元(native gate)
# [X90 @ Pauli @ X90dag によって'I', 'X', 'Y', 'Z'が変換される先], [変換後のそれぞれの符号]
X90 = [['I', 'X', 'Z', 'Y'], [1, 1, 1, -1]] 
Z90 = [['I', 'Y', 'X', 'Z'], [1, 1, -1, 1]]

    
# 変換同士の積の定義
def compose_1Q(transform_first, transform_second):
    
    transform_new = [['I', 'X', 'Y', 'Z'], [1, 1, 1, 1]]
    
    I_idx = transform_first[0].index('I') #'I'のある列番号
    X_idx = transform_first[0].index('X')
    Y_idx = transform_first[0].index('Y')
    Z_idx = transform_first[0].index('Z')
    
    transform_new[0][I_idx] = transform_second[0][0] #0行目リストの変換
    transform_new[0][X_idx] = transform_second[0][1]
    transform_new[0][Y_idx] = transform_second[0][2]
    transform_new[0][Z_idx] = transform_second[0][3]
    
    transform_new[1][I_idx] = transform_first[1][I_idx] * transform_second[1][0] #1行目リストの変換
    transform_new[1][X_idx] = transform_first[1][X_idx] * transform_second[1][1]
    transform_new[1][Y_idx] = transform_first[1][Y_idx] * transform_second[1][2]
    transform_new[1][Z_idx] = transform_first[1][Z_idx] * transform_second[1][3]
    
    return transform_new
    

# 1qubitクリフォードゲートの探索(全24種類)

# ゲート個数がgate_numの時の, それぞれのゲートをX90, Z90どちらかから選ぶ
clifford_list_1Q = []
total_gate_num = 10 #最大ゲート数

for gate_num in range(2 ** total_gate_num): #全ゲート個数を0〜total_gate_numまで掃引

    gate_list = [] #[['I', 'X', 'Z', 'Y'], [1, 1, 1, -1]]などのリストを格納
    gate_name_list = [] #'X90'などのゲート名のリストを格納
    
    while gate_num > 1:

        rem = gate_num % 2 #余り
        if rem == 0: #偶数ならX90を追加
            gate_list.append(X90)
            gate_name_list.append('X90')
        else: #奇数ならZ90を追加
            gate_list.append(Z90)
            gate_name_list.append('Z90')

        gate_num = gate_num // 2 #商, 2進数の次の桁を探索する
    
    
    transform = [['I', 'X', 'Y', 'Z'], [1, 1, 1, 1]] #初期ゲート(identity)
    for gate in gate_list:
        transform = compose_1Q(transform, gate) #gate_listに従ってゲートを変換していく
    
    clifford_list_col0 = [row[0] for row in clifford_list_1Q] #clifford_listの0列目
    clifford_list_col1 = [row[1] for row in clifford_list_1Q] #clifford_listの1列目
    
    if transform not in clifford_list_col1: #1列目だけのリストにtransformがなければgate_name_listと共に追加
        clifford_list_1Q.append([gate_name_list, transform])
    else:
        idx = clifford_list_col1.index(transform) #clifford_list_col1に既にあったtransformに等しい変換の要素番号
        X90_num = clifford_list_col0[idx].count('X90') #それに対応するgate_name中の'X90'の個数の取り出し
        X90_num_new = gate_name_list.count('X90') #新規gate_name中の'X90'の個数の取り出し
        
        #同じゲートならX90が少ない実現法を採用する. Z90はvirtual-Zでゼロ時間ゼロ誤差で実現できるのでいくら含んでいてもよい
        if X90_num > X90_num_new: 
            clifford_list_1Q[idx] = [gate_name_list, transform] #置き換え
    

def make_rand_clifford_list_1Q(
        n_gate: int,
        interleaved_gate: str | None = None
        ) -> list:
    """
    1qubitのクリフォード群からランダムにn_gate個のクリフォードゲートを選び, 
    interleaved_gateに入力があれば1つおきにinterleaved_gateを挿入し,
    さらに最後尾に全体の逆行列となるクリフォードゲートを加えた, n_gate+1列 or 2*n_gate+1列のリストを作成する.

    interleaved_gateは
    [['X90'], [['I', 'X', 'Z', 'Y'], [1, 1, 1, -1]]] や
    [['Z90'], [['I', 'Y', 'X', 'Z'], [1, 1, -1, 1]]] のような形で指定する. 

    Parameters
    ----------
    n_gate : int
        クリフォードゲートの個数.
    interleaved_gate : list
        挿入するターゲットゲート.
    
    Returns
    -------
    rand_clifford_list : list
        ランダムに選ばれたn_gate個のクリフォードゲート (+ interleaved_gateの挿入) + 全体の逆行列クリフォードゲートの, n_gate+1列 or 2*n_gate+1列のリスト.
    """

    #クリフォードゲートL個をランダムに抽出, 間にinterleaved_gateを挿入
    rand_clifford_list = []
    
    if interleaved_gate is None:
        for i in range(n_gate):
            rand_clifford = random.choices(clifford_list_1Q, k=1)
            rand_clifford_list.append(rand_clifford[0])
    else:
        for i in range(2*n_gate):
            if i%2==0:
                rand_clifford = random.choices(clifford_list_1Q, k=1)
                rand_clifford_list.append(rand_clifford[0])
            else:
                rand_clifford_list.append(interleaved_gate)

    #初期ゲート(identity)
    transform = [['I', 'X', 'Y', 'Z'], [1, 1, 1, 1]] 

    #rand_clifford_listに従って, identityからゲートを変換していく
    for clifford in rand_clifford_list:
        compose_1Q(transform, clifford[1]) 
    
    # トータルのゲートの逆行列を求める
    for clifford in clifford_list_1Q:

        transform_test = compose_1Q(transform, clifford[1])

        #identityに戻れば採用
        if transform_test == [['I', 'X', 'Y', 'Z'], [1, 1, 1, 1]]: 
            last_clifford = clifford
            break
    
        if i == len(clifford_list_1Q)-1: #identityに戻らなければエラー
            raise Exception('Error: Cannot find the Clifford gate in column gate_num + 1 such that the total transformation is identity.')
    
    rand_clifford_list.append(last_clifford)

    return rand_clifford_list



# 2qubit RBに必要な関数の定義    
    
def compose_2Q(transform_first, transform_second):
    """
    変換同士の積の定義
    第1引数のtransform_firstと第2引数のtransform_secondの積が返される. 

    Parameters
    ----------
    transform_first: list
        変換行列1
    transform_second: list
        変換行列2

    Returns
    -------
    transform_new: list
        変換行列1と変換行列2の積
    """
    
    transform_new = [
        ['II','IX','IY','IZ', 'XI','XX','XY','XZ', 'YI','YX','YY','YZ', 'ZI','ZX','ZY','ZZ'], 
        [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    ] 
    
    II_idx = transform_first[0].index('II') #'I'のある列番号
    IX_idx = transform_first[0].index('IX')
    IY_idx = transform_first[0].index('IY')
    IZ_idx = transform_first[0].index('IZ')
    
    XI_idx = transform_first[0].index('XI') 
    XX_idx = transform_first[0].index('XX')
    XY_idx = transform_first[0].index('XY')
    XZ_idx = transform_first[0].index('XZ')
    
    YI_idx = transform_first[0].index('YI') 
    YX_idx = transform_first[0].index('YX')
    YY_idx = transform_first[0].index('YY')
    YZ_idx = transform_first[0].index('YZ')
    
    ZI_idx = transform_first[0].index('ZI') 
    ZX_idx = transform_first[0].index('ZX')
    ZY_idx = transform_first[0].index('ZY')
    ZZ_idx = transform_first[0].index('ZZ')
    
    transform_new[0][II_idx] = transform_second[0][0] #0行目リストの変換
    transform_new[0][IX_idx] = transform_second[0][1]
    transform_new[0][IY_idx] = transform_second[0][2]
    transform_new[0][IZ_idx] = transform_second[0][3]
    
    transform_new[0][XI_idx] = transform_second[0][4] 
    transform_new[0][XX_idx] = transform_second[0][5]
    transform_new[0][XY_idx] = transform_second[0][6]
    transform_new[0][XZ_idx] = transform_second[0][7]
    
    transform_new[0][YI_idx] = transform_second[0][8] 
    transform_new[0][YX_idx] = transform_second[0][9]
    transform_new[0][YY_idx] = transform_second[0][10]
    transform_new[0][YZ_idx] = transform_second[0][11]
    
    transform_new[0][ZI_idx] = transform_second[0][12]
    transform_new[0][ZX_idx] = transform_second[0][13]
    transform_new[0][ZY_idx] = transform_second[0][14]
    transform_new[0][ZZ_idx] = transform_second[0][15]
    
    transform_new[1][II_idx] = transform_first[1][II_idx] * transform_second[1][0] #1行目リストの変換
    transform_new[1][IX_idx] = transform_first[1][IX_idx] * transform_second[1][1]
    transform_new[1][IY_idx] = transform_first[1][IY_idx] * transform_second[1][2]
    transform_new[1][IZ_idx] = transform_first[1][IZ_idx] * transform_second[1][3]
    
    transform_new[1][XI_idx] = transform_first[1][XI_idx] * transform_second[1][4]
    transform_new[1][XX_idx] = transform_first[1][XX_idx] * transform_second[1][5]
    transform_new[1][XY_idx] = transform_first[1][XY_idx] * transform_second[1][6]
    transform_new[1][XZ_idx] = transform_first[1][XZ_idx] * transform_second[1][7]
    
    transform_new[1][YI_idx] = transform_first[1][YI_idx] * transform_second[1][8]
    transform_new[1][YX_idx] = transform_first[1][YX_idx] * transform_second[1][9]
    transform_new[1][YY_idx] = transform_first[1][YY_idx] * transform_second[1][10]
    transform_new[1][YZ_idx] = transform_first[1][YZ_idx] * transform_second[1][11]
    
    transform_new[1][ZI_idx] = transform_first[1][ZI_idx] * transform_second[1][12]
    transform_new[1][ZX_idx] = transform_first[1][ZX_idx] * transform_second[1][13]
    transform_new[1][ZY_idx] = transform_first[1][ZY_idx] * transform_second[1][14]
    transform_new[1][ZZ_idx] = transform_first[1][ZZ_idx] * transform_second[1][15]

    return transform_new


# 2Q_clifford_listをロード
import os, pickle
dir_path = os.path.dirname(__file__)
path = os.path.join(dir_path, "2Q_clifford_list_ZX.bin")
with open(path, "rb") as p:
    clifford_list_2Q = pickle.load(p)


def make_rand_clifford_list_2Q(
        n_gate: int,
        interleaved_gate: str | None = None
        ) -> list:
    """
    1qubitのクリフォード群からランダムにn_gate個のクリフォードゲートを選び, 
    interleaved_gateに入力があれば1つおきにinterleaved_gateを挿入し,
    さらに最後尾に全体の逆行列となるクリフォードゲートを加えた, n_gate+1列 or 2*n_gate+1列のリストを作成する.

    interleaved_gateは
    [
        ['ZX90'], 
        [
            ['II','IX','ZZ','ZY', 'YX','YI','XY','XZ', 'XX','XI','YY','YZ', 'ZI','ZX','IZ','IY'], 
            [1,1,1,-1, 1,1,1,1, -1,-1,1,1, 1,1,1,-1],
        ]
    ]
    のような形で指定する. 

    Parameters
    ----------
    n_gate : int
        クリフォードゲートの個数.
    interleaved_gate : list
        挿入するターゲットゲート.
    
    Returns
    -------
    rand_clifford_list : list
        ランダムに選ばれたn_gate個のクリフォードゲート (+ interleaved_gateの挿入) + 全体の逆行列クリフォードゲートの, n_gate+1列 or 2*n_gate+1列のリスト.
    """

    flag = True
    
    while flag:
        
        rand_clifford_list = []
        
        if interleaved_gate is None:
            for i in range(n_gate):
                rand_clifford = random.choices(clifford_list_2Q, k=1)
                rand_clifford_list.append(rand_clifford[0])
        else:
            for i in range(2*n_gate):
                if i%2==0:
                    rand_clifford = random.choices(clifford_list_2Q, k=1)
                    rand_clifford_list.append(rand_clifford[0])
                else:
                    rand_clifford_list.append(interleaved_gate)

        transform = [
            ['II','IX','IY','IZ', 'XI','XX','XY','XZ', 'YI','YX','YY','YZ', 'ZI','ZX','ZY','ZZ'], 
            [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
        ] #初期ゲート(identity)

        for clifford in rand_clifford_list:
            transform = compose_2Q(transform, clifford[1]) #rand_clifford_listに従って, identityからゲートを変換していく
        # print(f"トータルの変換: {transform}")

        """トータルのゲートの逆行列を求める"""
        for clifford in clifford_list_2Q:

            transform_test = compose_2Q(transform, clifford[1])
            # print(f"transform_test = {transform_test}")

            if transform_test == [
                ['II','IX','IY','IZ', 'XI','XX','XY','XZ', 'YI','YX','YY','YZ', 'ZI','ZX','ZY','ZZ'], 
                [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            ]: #identityに戻れば採用
                last_clifford = clifford
                flag = False
                # print(f"last_clifford = {last_clifford}")
                break
            

    rand_clifford_list.append(last_clifford)
    # print(f"rand_clifford_list = {rand_clifford_list}")
    
    return rand_clifford_list


# フィッティングとゲート忠実度の計算
def get_average_fiderity_for_RB(
    gate_num_list,
    result_list_list,
    graph=True,
    ):
    """
    Randomized benchmarkingにおいて平均忠実度を求める関数.

    Parameters
    ----------
    gate_num_list : list
        Cliffordゲート数のリスト.
    result_list_list : list
        各クリフォードゲート数における実験結果のリストのリスト.
    graph : bool
        グラフを描画するかどうか. デフォルトはTrue.

    Returns
    -------
    avg_fidelity : float
        平均忠実度.
    """
    avg_list=[]
    std_list=[]
    for idx, gate_num in enumerate(gate_num_list):
        data_avg = np.average(result_list_list[idx])
        avg_list.append(data_avg)
        data_std = np.std(result_list_list[idx])
        std_list.append(data_std)

    def rb_func(x, a, p, b):
        return a * p**x + b

    coef, cov = curve_fit(rb_func, gate_num_list, avg_list, p0=[1, 0.99, 0], maxfev = 100000)

    #分極解消度
    a_fit = coef[0]
    p_fit = coef[1]
    b_fit = coef[2]
    print(f"分極解消度: {p_fit}")

    #平均ゲートエラー率
    avg_err = (2-1)/2 * (1 - p_fit)
    print(f"平均ゲートエラー率: {avg_err}")

    #平均忠実度
    avg_fidelity = (1+p_fit)/2
    print(f"平均ゲート忠実度: {avg_fidelity}")

    # グラフ描画
    if graph:
        plt.figure(figsize=(6, 4))

        # 測定結果のプロット
        plt.violinplot(result_list_list, positions=gate_num_list, widths=1)
        # for i, gate_num in enumerate(gate_num_list):
        #     plt.plot([gate_num]*type_num, result_list[i], 'o')
        plt.scatter(gate_num_list, avg_list, color="blue", label="average")

        #フィッティングのプロット
        xs = np.linspace(0, gate_num_list[-1], 1000)
        plt.plot(xs, a_fit * p_fit**xs + b_fit, color="blue", label="fitting")
                
        plt.xlabel('Number of random Clifford gates')
        plt.ylabel('Z expectation value')
        plt.title(f'Randomized benchmarking')
        plt.grid()
        plt.legend()
        # plt.ylim(-1, 1)
        # plt.semilogx()
        plt.show()
    
    return avg_fidelity
    
