# -*- coding: utf-8 -*-
"""
FBCCA (FBSCCA) 评估与调参脚本。

FBCCA 是无训练 (calibration-free) 的 SSVEP 解码算法：
- 生成正弦余弦参考信号 (Yf) 与多通道 EEG 做 CCA
- 滤波器组将信号分解为多个子带
- 子带相关系数加权求和，最大值对应频率即为预测结果

可调参数:
    n_harmonics : int        — 谐波数 (1-5)
    n_subbands  : int        — 子带数 (1-5)
    filter_order : int       — 滤波器阶数
    a, b        : float      — 子带权重公式: (i+1)^(-a) + b

用法:
    python eval_fbcca.py                          # 默认参数评估
    python eval_fbcca.py --tune                   # 网格搜索最优参数
    python eval_fbcca.py --compare                # 与 FBTDCA 模型对比
"""
import os, sys, glob, argparse
import numpy as np
from math import log2
from itertools import product

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, '..'))

from metabci.brainda.algorithms.decomposition.cca import FBSCCA
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank,
    generate_cca_references,
)

# ========== 配置 ==========
SAMPLE_RATE = 250
STIM_FREQS = [8.25, 11.0, 13.75, 16.5]
STIM_PHASES = [0.0, 0.0, 0.0, 0.0]  # 无相位偏移
WINDOW_LENGTHS = [125, 250, 375, 500]  # 采样点
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]
N_CLASSES = 4
DATA_ROOT = os.path.join(_BASE, 'data_self_test')


# ========== 数据加载 ==========
def load_data(root, occipital_indices=OCCIPITAL_INDICES):
    X, y = [], []
    for label in range(N_CLASSES):
        folder = os.path.join(root, str(label + 1))
        if not os.path.isdir(folder):
            continue
        files = (glob.glob(os.path.join(folder, '*offset000.npy')) +
                 glob.glob(os.path.join(folder, 'browser_trial_*.npy')))
        for f in sorted(files):
            data = np.load(f)
            if data.shape[0] == 15:           # 14 EEG + Trigger
                data = data[:14, :]
            data = data[occipital_indices, :]  # (8, samples)
            X.append(data)
            y.append(label)
    return np.array(X), np.array(y)


# ========== ITR ==========
def wolpaw_itr(N, P, T):
    if P <= 0: return 0.0
    if P >= 0.9999: P = 0.9999
    if P <= 1.0 / N: return 0.0
    B = log2(N) + P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))
    return B * (60.0 / T)


# ========== FBCCA 评估 ==========
def evaluate_fbcca(X, y, window_len, *,
                   n_harmonics=3, n_subbands=5,
                   a=1.25, b=0.25, filter_order=6):
    """
    参数:
        n_harmonics : 谐波数
        n_subbands  : 滤波器组子带数
        a, b        : 子带权重 = (i+1)^(-a) + b
    """
    # ---- 生成参考信号 Yf ----
    Yf = generate_cca_references(
        freqs=STIM_FREQS,
        phases=STIM_PHASES,
        srate=SAMPLE_RATE,
        T=window_len / SAMPLE_RATE,
        n_harmonics=n_harmonics,
    )  # (n_classes, 2*n_harmonics, n_samples)

    # ---- 生成滤波器组 ----
    # 经典 FBCCA 子带: [m*8, 90] Hz, m=1,2,...,n_subbands
    # 参考: Chen et al., JNE 2015
    wp = [(8 * (i + 1), 90) for i in range(n_subbands)]
    ws = [(max(4, 8 * (i + 1) - 2), min(92, 92)) for i in range(n_subbands)]
    filterbank = generate_filterbank(
        passbands=wp, stopbands=ws, srate=SAMPLE_RATE, order=filter_order, rp=0.5
    )
    filterweights = np.array([(i + 1) ** (-a) + b for i in range(n_subbands)])

    # ---- 构建 FBSCCA ----
    clf = FBSCCA(
        filterbank=filterbank,
        n_components=1,
        filterweights=filterweights,
        n_jobs=1,
    )

    # ---- 截取窗口 ----
    X_win = X[:, :, :window_len]  # (n_trials, n_chans, window_len)
    X_win = X_win - np.mean(X_win, axis=-1, keepdims=True)

    # FBCCA 不需要训练数据，但 fit() 接口需要传 X,y,Yf
    clf.fit(X_win, y, Yf=Yf)
    pred = clf.predict(X_win)
    acc = np.mean(pred == y)
    itr = wolpaw_itr(N_CLASSES, acc, window_len / SAMPLE_RATE)
    return acc, itr


# ========== 网格搜索 ==========
def grid_search(X, y, window_len=500):
    best_acc, best_itr, best_params = 0, 0, {}
    results = []

    harmonics = [1, 2, 3, 4, 5]
    subbands  = [2, 3, 4, 5]
    a_vals    = [0.5, 0.75, 1.0, 1.25, 1.5]
    b_vals    = [0.0, 0.1, 0.25, 0.5]

    for nh, ns, a, b in product(harmonics, subbands, a_vals, b_vals):
        acc, itr = evaluate_fbcca(X, y, window_len,
                                  n_harmonics=nh, n_subbands=ns, a=a, b=b)
        results.append({
            'n_harmonics': nh, 'n_subbands': ns, 'a': a, 'b': b,
            'acc': acc, 'itr': itr,
        })
        if acc > best_acc:
            best_acc, best_itr = acc, itr
            best_params = {'nh': nh, 'ns': ns, 'a': a, 'b': b}

    return results, best_acc, best_itr, best_params


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description='FBCCA 评估与调参')
    parser.add_argument('--tune', action='store_true', help='网格搜索最优参数')
    parser.add_argument('--compare', action='store_true', help='与 FBTDCA 模型对比')
    args = parser.parse_args()

    X, y = load_data(DATA_ROOT)
    print(f'数据: {len(X)} trials, 类别分布 {np.bincount(y)}')
    print(f'通道: {X.shape[1]} ({X.shape[2]} samples = {X.shape[2]/SAMPLE_RATE*1000:.0f}ms)')
    print()

    if args.tune:
        print('=' * 60)
        print('网格搜索 (2.0s 窗口)')
        print('=' * 60)
        results, best_acc, best_itr, best = grid_search(X, y, window_len=500)

        # Top 5
        results.sort(key=lambda r: r['acc'], reverse=True)
        print(f'\nTop 5 参数组合 ({len(results)} total):')
        print(f'{"谐波":>6} {"子带":>6} {"a":>6} {"b":>6} {"准确率":>10} {"ITR":>10}')
        for r in results[:5]:
            print(f'{r["n_harmonics"]:>6} {r["n_subbands"]:>6} {r["a"]:>6.2f} {r["b"]:>6.2f} '
                  f'{r["acc"]*100:>9.1f}% {r["itr"]:>9.1f}')
        print(f'\n最优: Nh={best["nh"]} Ns={best["ns"]} a={best["a"]} b={best["b"]} '
              f'→ {best_acc*100:.1f}% / {best_itr:.1f} bps')

    elif args.compare:
        print('=' * 60)
        print('FBCCA vs FBTDCA 对比')
        print('=' * 60)

        import joblib
        MODEL_DIR = os.path.join(_BASE, 'models', 'browser')
        GW_MODEL_PATHS = {
            125: os.path.join(MODEL_DIR, 'model_125_browser.pkl'),
            250: os.path.join(MODEL_DIR, 'model_250_browser.pkl'),
            375: os.path.join(MODEL_DIR, 'model_375_browser.pkl'),
            500: os.path.join(MODEL_DIR, 'self_ssvep_model_browser.pkl'),
        }

        print(f'\n{"窗口":>10} {"FBCCA":>12} {"FBTDCA":>12} {"FBCCA ITR":>12} {"FBTDCA ITR":>12}')
        print('-' * 60)
        for wl in WINDOW_LENGTHS:
            # FBCCA (默认参数)
            fbcca_acc, fbcca_itr = evaluate_fbcca(X, y, wl)

            # FBTDCA (从模型文件)
            model = joblib.load(GW_MODEL_PATHS[wl])
            X_win = X[:, :, :wl]
            fbd_pred = np.array([np.argmax(model.transform(d[np.newaxis, ...])[0])
                                  for d in X_win])
            fbd_acc = np.mean(fbd_pred == y)
            fbd_itr = wolpaw_itr(N_CLASSES, fbd_acc, wl / SAMPLE_RATE)

            print(f'{wl}pt/{wl*4}ms {fbcca_acc*100:>11.1f}% {fbd_acc*100:>11.1f}% '
                  f'{fbcca_itr:>11.1f} {fbd_itr:>11.1f}')

    else:
        # 默认: 全窗口 + 默认参数评估
        print('=' * 60)
        print('FBCCA 各窗口评估 (默认参数: Nh=3 Ns=5 a=1.25 b=0.25)')
        print('=' * 60)
        for wl in WINDOW_LENGTHS:
            acc, itr = evaluate_fbcca(X, y, wl)
            print(f'  {wl}点 ({wl/SAMPLE_RATE*1000:.0f}ms):  {acc*100:.1f}%  '
                  f'ITR={itr:.1f} bps')

        print()
        print('=' * 60)
        print('谐波数扫描 (2.0s 窗口, Ns=5)')
        print('=' * 60)
        for nh in [1, 2, 3, 4, 5]:
            acc, itr = evaluate_fbcca(X, y, 500, n_harmonics=nh)
            print(f'  Nh={nh}:  {acc*100:.1f}%  ITR={itr:.1f} bps')

        print()
        print('=' * 60)
        print('子带数扫描 (2.0s 窗口, Nh=3)')
        print('=' * 60)
        for ns in [1, 2, 3, 4, 5]:
            acc, itr = evaluate_fbcca(X, y, 500, n_subbands=ns)
            print(f'  Ns={ns}:  {acc*100:.1f}%  ITR={itr:.1f} bps')

    print('\n完成。')


if __name__ == '__main__':
    main()
