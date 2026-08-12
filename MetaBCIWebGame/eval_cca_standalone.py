# -*- coding: utf-8 -*-
"""
独立 FBCCA 评估脚本 — 自实现 CCA，绕过 metabci SCCA 的数值稳定性问题。

CCA 使用 SVD 分解: max_corr = max(svd(Cxx^{-1/2} @ Cxy @ Cyy^{-1/2}))
"""
import os, sys, glob, argparse
import numpy as np
from math import log2
from scipy.signal import iircomb, filtfilt, cheby1, sosfiltfilt

# ========== 配置 ==========
SAMPLE_RATE = 250
STIM_FREQS = np.array([8.25, 11.0, 13.75, 16.5])
N_CLASSES = 4
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_self_test')


# ========== 自实现 CCA ==========
def cca_corr(X, Y, reg=1e-6):
    """计算 X (n_ch, n_t) 与 Y (n_ref, n_t) 的最大典型相关系数。

    返回: max_corr (float)
    """
    n_ch, n_t = X.shape
    n_ref = Y.shape[0]
    # Demean
    X_c = X - X.mean(axis=1, keepdims=True)
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    # 协方差矩阵
    Cxx = X_c @ X_c.T / (n_t - 1)
    Cyy = Y_c @ Y_c.T / (n_t - 1)
    Cxy = X_c @ Y_c.T / (n_t - 1)
    # 正则化
    Cxx += reg * np.eye(n_ch)
    Cyy += reg * np.eye(n_ref)
    # Cxx^{-1/2} 和 Cyy^{-1/2}
    Ux, Sx, _ = np.linalg.svd(Cxx)
    Uy, Sy, _ = np.linalg.svd(Cyy)
    Cxx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(np.maximum(Sx, 1e-10))) @ Ux.T
    Cyy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(np.maximum(Sy, 1e-10))) @ Uy.T
    # 白化后的互协方差
    K = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    _, S, _ = np.linalg.svd(K)
    return S[0] if len(S) > 0 else 0.0


def fbcca_predict_one(X_trial, Yf_list, filterbank_sos, filterweights,
                       n_harmonics=3, reg=1e-6):
    """对单个试次做 FBCCA 预测。

    X_trial: (n_ch, n_samples) — 单试次 EEG
    Yf_list: list of (2*n_harmonics, n_samples) — 各频率参考信号
    filterbank_sos: list of SOS filter coefficients
    filterweights: (n_subbands,)
    """
    n_subbands = len(filterbank_sos)
    n_freqs = len(Yf_list)
    all_rhos = np.zeros((n_subbands, n_freqs))

    for sb in range(n_subbands):
        # 子带滤波
        X_filt = sosfiltfilt(filterbank_sos[sb], X_trial, axis=-1)
        for fi in range(n_freqs):
            rho = cca_corr(X_filt, Yf_list[fi], reg=reg)
            all_rhos[sb, fi] = rho

    # 子带加权组合
    weighted = np.sum(all_rhos * filterweights[:, np.newaxis], axis=0)
    return np.argmax(weighted)


def fbcca_predict_batch(X, Yf_list, filterbank_sos, filterweights,
                        n_harmonics=3, reg=1e-6, verbose=False):
    """批量 FBCCA 预测。"""
    preds = []
    for i in range(len(X)):
        pred = fbcca_predict_one(
            X[i], Yf_list, filterbank_sos, filterweights,
            n_harmonics=n_harmonics, reg=reg)
        preds.append(pred)
        if verbose and (i + 1) % 20 == 0:
            print(f'  ... {i+1}/{len(X)}')
    return np.array(preds)


# ========== 参考信号生成 ==========
def generate_ref_signal(freqs, phase, srate, n_samples, n_harmonics):
    """生成多频率 sine-cosine 参考信号。

    返回: list of (2*n_harmonics, n_samples)
    """
    t = np.arange(n_samples) / srate
    Yf_list = []
    for f, phi in zip(freqs, phase):
        ref = []
        for h in range(1, n_harmonics + 1):
            ref.append(np.sin(2 * np.pi * f * h * t + phi * h))
            ref.append(np.cos(2 * np.pi * f * h * t + phi * h))
        Yf_list.append(np.array(ref))
    return Yf_list


# ========== 滤波器组生成 ==========
def generate_filterbank_sos(n_subbands, srate, order=4):
    """生成 FBCCA 标准滤波器组。

    n_subbands: 子带数
    返回: list of SOS arrays
    """
    sos_list = []
    for i in range(n_subbands):
        wpl = 8 * (i + 1)       # 低通
        wph = 90                 # 高通固定
        sos = cheby1(order, 0.5, [wpl, wph], btype='bandpass',
                     output='sos', fs=srate)
        sos_list.append(sos)
    return sos_list


# ========== ITR ==========
def wolpaw_itr(N, P, T):
    if P <= 0 or P >= 0.9999:
        P = 0.9999 if P >= 0.9999 else P
    if P <= 1.0 / N:
        return 0.0
    B = log2(N) + P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))
    return B * (60.0 / T)


# ========== 数据加载 ==========
def load_data(root):
    X, y = [], []
    for label in range(N_CLASSES):
        folder = os.path.join(root, str(label + 1))
        if not os.path.isdir(folder):
            continue
        files = (glob.glob(os.path.join(folder, '*offset000.npy')) +
                 glob.glob(os.path.join(folder, 'browser_trial_*.npy')))
        for f in sorted(files):
            data = np.load(f)
            if data.shape[0] == 15:
                data = data[:14, :]
            # 枕区 8 通道
            data = data[[2, 3, 4, 5, 6, 7, 8, 9], :]
            X.append(data)
            y.append(label)
    return np.array(X), np.array(y)


# ========== 预处理 ==========
def preprocess(X, srate=250):
    """50Hz 陷波 + 通道去均值，对齐 FBTDCA 预处理管线。"""
    b_notch, a_notch = iircomb(50, 35, ftype='notch', fs=srate)
    for i in range(len(X)):
        X[i] = filtfilt(b_notch, a_notch, X[i], axis=-1)
        X[i] = X[i] - X[i].mean(axis=1, keepdims=True)
    return X


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='网格搜索')
    parser.add_argument('--ws', type=float, default=2.0, help='窗口时长(秒)')
    args = parser.parse_args()

    X, y = load_data(DATA_ROOT)
    X = preprocess(X)
    print(f'数据: {len(X)} trials, {X.shape[1]}ch x {X.shape[2]}samples')
    print(f'类别分布: {np.bincount(y)}')
    print()

    STIM_PHASES = np.array([0, 0, 0, 0])

    if args.tune:
        print('=' * 60)
        print('网格搜索 (FBCCA 超参数)')
        print('=' * 60)
        n_samples = int(args.ws * SAMPLE_RATE)
        best = {'acc': 0}

        for nh in [1, 2, 3, 4, 5]:
            for ns in [2, 3, 4, 5]:
                for a in [0.5, 1.0, 1.25, 1.5]:
                    for b_s in [0.0, 0.1, 0.25, 0.5]:
                        sos = generate_filterbank_sos(ns, SAMPLE_RATE, order=4)
                        fw = np.array([(i+1)**(-a) + b_s for i in range(ns)])
                        Yf_list = generate_ref_signal(
                            STIM_FREQS, STIM_PHASES, SAMPLE_RATE,
                            n_samples, nh)
                        X_win = X[:, :, :n_samples].copy()
                        preds = fbcca_predict_batch(
                            X_win, Yf_list, sos, fw, nh, reg=1e-4)
                        acc = np.mean(preds == y)
                        itr = wolpaw_itr(N_CLASSES, acc, n_samples/SAMPLE_RATE)
                        if acc > best['acc']:
                            best = {'nh': nh, 'ns': ns, 'a': a, 'b': b,
                                    'acc': acc, 'itr': itr}
                            print(f'  NEW BEST: Nh={nh} Ns={ns} a={a} b={b} '
                                  f'→ {acc*100:.1f}% / {itr:.1f} bps')
        print(f'\n最优: {best}')
    else:
        # 默认评估
        for ns_desc, n_subbands in [('标准CCA (无子带)', 1),
                                     ('FBCCA 5子带', 5)]:
            print('=' * 60)
            print(ns_desc)
            print('=' * 60)
            for wl_sec in [0.5, 1.0, 1.5, 2.0]:
                n_samples = int(wl_sec * SAMPLE_RATE)
                sos = generate_filterbank_sos(n_subbands, SAMPLE_RATE, order=4)
                fw = np.array([(i+1)**(-1.25) + 0.25 for i in range(n_subbands)])
                for nh in [1, 3]:
                    Yf_list = generate_ref_signal(
                        STIM_FREQS, STIM_PHASES, SAMPLE_RATE,
                        n_samples, nh)
                    X_win = X[:, :, :n_samples].copy()
                    preds = fbcca_predict_batch(
                        X_win, Yf_list, sos, fw, nh, reg=1e-4)
                    acc = np.mean(preds == y)
                    itr = wolpaw_itr(N_CLASSES, acc,  n_samples/SAMPLE_RATE)
                    label = f'{wl_sec}s Nh={nh}'
                    print(f'  {label:>12s}: acc={acc*100:5.1f}%  ITR={itr:6.1f} bps')
                print()

    print('完成。')


if __name__ == '__main__':
    main()
