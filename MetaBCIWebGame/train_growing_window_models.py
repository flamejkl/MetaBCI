# train_growing_window_models.py
# -*- coding: utf-8 -*-
"""
训练多个窗口长度的 FBTDCA 模型，用于 Growing Window 动态停止策略。
分别训练 0.5s (125点)、1.0s (250点)、1.5s (375点)、2.0s (500点) 四个模型。

复用 train_self_model 的核心函数，仅循环不同窗口长度。
"""
import sys
import os
_METABCI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metabci')
if os.path.isdir(os.path.join(_METABCI_ROOT, 'brainda')):
    sys.path.insert(0, os.path.abspath(os.path.join(_METABCI_ROOT, '..')))

import numpy as np
import joblib
from config import BASE_DIR
from train_self_model import (
    augment_ssvep_data, load_data_from_dirs, evaluate_model_kfold,
    AUG_NUM, AUG_DROP_PROB, AUG_NOISE_RATIO,
    AUG_SCALE_RANGE, AUG_ATTEN_RANGE,
)
from metabci.brainda.algorithms.decomposition.tdca import FBTDCA
from metabci.brainda.algorithms.decomposition.base import generate_cca_references
from metabci.brainda.algorithms.decomposition.base import generate_filterbank

DATA_ROOT = os.path.join(BASE_DIR, "data_self")
TARGET_FREQS = [8.25, 11.0, 13.75, 16.5]
SAMPLE_RATE = 250

MODEL_OUTPUTS = {
    125: os.path.join(BASE_DIR, "models", "browser", "model_125_browser.pkl"),
    250: os.path.join(BASE_DIR, "models", "browser", "model_250_browser.pkl"),
    375: os.path.join(BASE_DIR, "models", "browser", "model_375_browser.pkl"),
    500: os.path.join(BASE_DIR, "models", "browser", "self_ssvep_model_browser.pkl"),
}


def main():
    passbands = [[6, 90], [14, 90], [22, 90], [30, 90], [38, 90]]
    stopbands = [[4, 100], [10, 100], [16, 100], [24, 100], [32, 100]]
    filterbank = generate_filterbank(passbands, stopbands, SAMPLE_RATE)
    filterweights = [(i + 1) ** (-1.25) + 0.25 for i in range(len(passbands))]

    for L in [125, 250, 375, 500]:
        print(f"\n{'='*60}")
        print(f"训练窗口长度: {L} 点 ({L/SAMPLE_RATE:.1f} 秒)")
        print('='*60)

        X_raw, y_raw, groups_raw = load_data_from_dirs(
            DATA_ROOT, window_samples=L, use_occipital=True,
            offset_only=True,
        )
        print(f"数据形状: {X_raw.shape}, 分布: {np.bincount(y_raw)}")

        Yf = generate_cca_references(
            TARGET_FREQS, srate=SAMPLE_RATE, T=L/SAMPLE_RATE, n_harmonics=5,
        )

        mean_acc, std_acc = evaluate_model_kfold(
            FBTDCA, X_raw, y_raw, groups_raw,
            Yf=Yf, n_splits=5,
            filterbank=filterbank, filterweights=filterweights,
            n_components=3, padding_len=0,
        )
        print(f"FBTDCA 5折CV: {mean_acc:.4f} +/- {std_acc:.4f}")

        X_aug, y_aug = augment_ssvep_data(
            X_raw, y_raw,
            num_aug=AUG_NUM, drop_prob=AUG_DROP_PROB,
            noise_ratio=AUG_NOISE_RATIO,
            scale_range=AUG_SCALE_RANGE, atten_range=AUG_ATTEN_RANGE,
            random_seed=2025,
        )
        model = FBTDCA(
            filterbank=filterbank, filterweights=filterweights,
            n_components=3, padding_len=0,
        )
        model.fit(X_aug, y_aug, Yf=Yf)
        joblib.dump(model, MODEL_OUTPUTS[L])
        print(f"模型已保存: {MODEL_OUTPUTS[L]}")

    print("\n所有窗口长度模型训练完成！")


if __name__ == "__main__":
    main()
