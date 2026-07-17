# eval_itr.py
# -*- coding: utf-8 -*-
"""
信息传输率 (Information Transfer Rate, ITR) 评估脚本。

基于 Wolpaw ITR 公式 (Wolpaw et al., 2002):
    B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))
    ITR = B * (60 / T)

其中 N 为类别数，P 为准确率，T 为平均决策时间（秒）。

用法:
    python eval_itr.py                        # 使用默认4模型Growing Window评估
    python eval_itr.py --mode fixed           # 固定2s窗口评估
    python eval_itr.py --mode gw --data_root data_self_test
"""
import os
import sys
# Ensure metabci framework is on path
_METABCI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metabci')
if os.path.isdir(os.path.join(_METABCI_ROOT, 'brainda')):
    sys.path.insert(0, os.path.abspath(os.path.join(_METABCI_ROOT, '..')))

import argparse
import numpy as np
from math import log2
from collections import deque
from config import (
    SAMPLE_RATE, GW_MODEL_PATHS, OCCIPITAL_INDICES, BASE_DIR
)
from metabci.brainda.algorithms.decomposition import GrowingWindowDecoder


# ============================================================
#  ITR 核心公式
# ============================================================
def wolpaw_itr(N, P, T):
    """Wolpaw ITR (bits/min).

    Parameters
    ----------
    N : int   – 类别数
    P : float – 分类准确率 (0~1)
    T : float – 平均决策时间 (秒)

    Returns
    -------
    itr : float – 信息传输率 (bits/min)
    """
    if P <= 0 or P >= 1:
        return 0.0
    if P <= 1.0 / N:
        return 0.0  # 低于随机水平，不计算
    B = log2(N) + P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))
    return B * (60.0 / T)


def compute_itr(decisions, labels, times, n_classes=4):
    """从一批试次的解码结果计算 ITR。

    Parameters
    ----------
    decisions : list[int]
        每个试次的预测类别
    labels : list[int]
        每个试次的真实类别
    times : list[float]
        每个试次的决策时间 (秒)
    n_classes : int
        类别数

    Returns
    -------
    dict with keys: accuracy, avg_time, itr, total_trials, early_stop_rate
    """
    correct = sum(1 for d, l in zip(decisions, labels) if d == l)
    total = len(decisions)
    acc = correct / total if total > 0 else 0.0
    avg_t = np.mean(times) if times else 0.0

    itr = wolpaw_itr(n_classes, acc, avg_t)

    early = sum(1 for t in times if t < 2.0)
    early_rate = early / total if total > 0 else 0.0

    return {
        'accuracy': acc,
        'avg_time_s': avg_t,
        'itr_bits_per_min': round(itr, 2),
        'total_trials': total,
        'correct_trials': correct,
        'early_stop_rate': early_rate,
        'early_stop_count': early,
    }


# ============================================================
#  评估主流程
# ============================================================
def load_test_data(data_root, occipital_indices=None):
    """加载测试数据，返回 (X_list, y_list)。"""
    import os, glob
    if occipital_indices is None:
        occipital_indices = OCCIPITAL_INDICES
    X, y = [], []
    for label in range(4):
        folder = os.path.join(data_root, str(label + 1))
        if not os.path.isdir(folder):
            continue
        for f in glob.glob(os.path.join(folder, '*offset000.npy')):
            data = np.load(f)
            data = data[occipital_indices, :]  # (8, 500)
            X.append(data)
            y.append(label)
    return X, y


def evaluate_growing_window(X, y):
    """Growing Window 动态停止策略评估。"""
    decoder = GrowingWindowDecoder(model_paths=GW_MODEL_PATHS)
    decisions, times, labels_list = [], [], []

    for data, label in zip(X, y):
        decoder.reset()
        decoder.reset_normaliser()  # 每试次独立，避免EMA跨试次污染
        decision = None
        for i in range(data.shape[1]):
            d, conf, t = decoder.feed(data[:, i])
            if d is not None:
                decision = d
                decisions.append(d)
                times.append(t)
                labels_list.append(label)
                break
        # 试次结束未决策 = 2s 未输出（理论上不会发生，因为有强制输出）
        if decision is None:
            decisions.append(-1)
            times.append(2.0)
            labels_list.append(label)

    return compute_itr(decisions, labels_list, times)


def evaluate_fixed_window(X, y, window_len=2.0):
    """固定 2s 窗口评估。"""
    from metabci.brainda.algorithms.decomposition import GrowingWindowDecoder
    decoder = GrowingWindowDecoder(model_paths=GW_MODEL_PATHS)
    decisions, times, labels_list = [], [], []

    n_samples = int(window_len * SAMPLE_RATE)
    for data, label in zip(X, y):
        decoder.reset()
        decoder.reset_normaliser()
        decision = None
        # 只取前 N 个样本
        window_data = data[:, :n_samples]
        for i in range(window_data.shape[1]):
            d, conf, t = decoder.feed(window_data[:, i])
            if d is not None:
                decision = d
                break
        if decision is not None:
            decisions.append(decision)
        else:
            decisions.append(-1)
        times.append(window_len)
        labels_list.append(label)

    return compute_itr(decisions, labels_list, times)


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='SSVEP ITR 评估')
    parser.add_argument('--mode', choices=['gw', 'fixed', 'both'], default='both',
                        help='评估模式: gw=Growing Window, fixed=固定窗口, both=两者')
    parser.add_argument('--data_root', type=str, default=None,
                        help='测试数据根目录 (默认: data_self_test)')
    parser.add_argument('--n_trials', type=int, default=0,
                        help='限制加载试次数 (0=全部)')
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(BASE_DIR, 'data_self_test')

    print(f"数据目录: {data_root}")
    X, y = load_test_data(data_root)

    # 限制试次数
    if args.n_trials > 0:
        X, y = X[:args.n_trials], y[:args.n_trials]

    print(f"加载试次: {len(X)} (类别分布: {np.bincount(y)})")
    print()

    if args.mode in ('gw', 'both'):
        print("=" * 60)
        print("Growing Window 动态停止策略")
        print("=" * 60)
        result = evaluate_growing_window(X, y)
        _print_result(result)

    if args.mode in ('fixed', 'both'):
        print()
        print("=" * 60)
        print("固定 2.0s 窗口")
        print("=" * 60)
        result = evaluate_fixed_window(X, y, window_len=2.0)
        _print_result(result)


def _print_result(r):
    print(f"  试次总数:      {r['total_trials']}")
    print(f"  正确数:        {r['correct_trials']}")
    print(f"  准确率:        {r['accuracy']*100:.2f}%")
    print(f"  平均决策时间:  {r['avg_time_s']*1000:.0f} ms")
    print(f"  提前停止率:    {r['early_stop_rate']*100:.1f}% ({r['early_stop_count']}/{r['total_trials']})")
    print(f"  ─────────────────────────────")
    print(f"  ITR:           {r['itr_bits_per_min']} bits/min")


if __name__ == '__main__':
    main()
