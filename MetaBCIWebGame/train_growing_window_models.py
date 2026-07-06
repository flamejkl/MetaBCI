# train_growing_window_models.py
# -*- coding: utf-8 -*-
"""
训练多个窗口长度的 FBTDCA 模型，用于 Growing Window 动态停止策略。
分别训练 0.5s (125点)、1.0s (250点)、1.5s (375点)、2.0s (500点) 四个模型。
复用原 train_self_model.py 的核心逻辑，但将 WINDOW_SAMPLES 作为参数传入。
"""
import os
import re
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from metabci.brainda.algorithms.decomposition.cca import FBTRCA
from metabci.brainda.algorithms.decomposition.tdca import FBTDCA
from fbcca_eigh import get_default_filterbank

# ============================================================
#  全局配置（保持不变）
# ============================================================
DATA_ROOT = r"D:\pycharm\PyCharm 2026.1\my-projects\MetaBCI\data_self"  # 仅使用 offset0 数据
TARGET_FREQS = [8.25, 11.0, 13.75, 16.5]
SAMPLE_RATE = 250
N_COMPONENTS_TDCA = 3
N_COMPONENTS_TRCA = 1
ENSEMBLE = True
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]   # 枕区通道

# 数据增强参数
AUG_NUM = 3
AUG_DROP_PROB = 0.5
AUG_NOISE_RATIO = 0.01
AUG_SCALE_RANGE = (0.9, 1.1)
AUG_ATTEN_RANGE = (0.4, 0.7)

# ============================================================
#  数据增强函数（同原版）
# ============================================================
def augment_ssvep_data(X, y, num_aug=3, drop_prob=0.5, noise_ratio=0.01,
                       scale_range=(0.9, 1.1), atten_range=(0.4, 0.7),
                       random_seed=None):
    rng = np.random.RandomState(random_seed)
    X_aug, y_aug = [], []
    n_trials, n_chans, n_samples = X.shape
    for i in range(n_trials):
        original = X[i]
        X_aug.append(original); y_aug.append(y[i])
        for _ in range(num_aug):
            aug = original.copy()
            scale = rng.uniform(*scale_range); aug *= scale
            if rng.rand() < drop_prob:
                ch = rng.randint(0, n_chans)
                atten = rng.uniform(*atten_range)
                aug[ch, :] *= atten
            sigma = noise_ratio * np.std(aug, axis=1, keepdims=True)
            aug = aug + rng.normal(0, sigma, aug.shape)
            X_aug.append(aug); y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

# ============================================================
#  数据加载（自动识别 offset 后缀）
# ============================================================
def load_data_from_dirs(root, window_samples, use_occipital=True):
    X_list, y_list, groups_list = [], [], []
    for label in range(4):
        dir_path = os.path.join(root, str(label+1))
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
        # 检查是否有 offset 后缀
        has_offset = any('offset' in f for f in files)
        if has_offset:
            # 如果存在 offset，只取 offset000（即起点对齐）
            files = [f for f in files if 'offset000' in f]
        # 否则 files 就是全部 .npy（它们本身就是 offset0）
        for fname in files:
            data = np.load(os.path.join(dir_path, fname))
            # 截取前 window_samples 个点（因为原始数据至少 500 点）
            if data.shape[1] < window_samples:
                print(f"警告：{fname} 长度 {data.shape[1]} < {window_samples}，跳过")
                continue
            data = data[:, :window_samples]
            X_list.append(data)
            y_list.append(label)
            match = re.search(r'hw_trial_(\d+)', fname)
            group_id = int(match.group(1)) if match else len(groups_list)
            groups_list.append(group_id)
    if not X_list:
        raise RuntimeError(f"未在 {root} 下找到任何数据文件！")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    groups = np.array(groups_list)
    if use_occipital:
        X = X[:, OCCIPITAL_INDICES, :]
        print(f"使用枕区 {len(OCCIPITAL_INDICES)} 个通道，新形状: {X.shape}")
    X = X - np.mean(X, axis=2, keepdims=True)
    X = X + 1e-10 * np.random.randn(*X.shape)
    return X, y, groups

def evaluate_model_kfold(model_class, X, y, groups, Yf=None, n_splits=5, **kwargs):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    fold = 0
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train_aug, y_train_aug = augment_ssvep_data(
            X_train_raw, y_train_raw,
            num_aug=AUG_NUM, drop_prob=AUG_DROP_PROB,
            noise_ratio=AUG_NOISE_RATIO,
            scale_range=AUG_SCALE_RANGE,
            atten_range=AUG_ATTEN_RANGE,
            random_seed=42+fold
        )
        if Yf is not None:
            model = model_class(**kwargs)
            model.fit(X_train_aug, y_train_aug, Yf=Yf)
        else:
            model = model_class(**kwargs)
            model.fit(X_train_aug, y_train_aug)
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        fold += 1
    return np.mean(accs), np.std(accs)

# ============================================================
#  主程序：循环训练不同长度
# ============================================================
def main():
    # 定义要训练的长度（采样点数）
    WINDOW_LENGTHS = [125, 250, 375, 500]  # 对应 0.5s, 1.0s, 1.5s, 2.0s
    # 对应的输出模型文件名
    model_names = {
        125: "model_125.pkl",
        250: "model_250.pkl",
        375: "model_375.pkl",
        500: "self_ssvep_model.pkl"  # 2s 仍使用原名，保持兼容
    }

    filterbank, filterweights = get_default_filterbank(SAMPLE_RATE)
    filterweights = np.array(filterweights)
    from metabci.brainda.algorithms.decomposition.base import generate_cca_references

    for L in WINDOW_LENGTHS:
        print(f"\n{'='*60}")
        print(f"训练窗口长度: {L} 点 ({L/SAMPLE_RATE:.1f} 秒)")
        print('='*60)
        # 加载数据（只加载 offset0，并截取前 L 点）
        X_raw, y_raw, groups_raw = load_data_from_dirs(DATA_ROOT, window_samples=L, use_occipital=True)
        print(f"数据形状: {X_raw.shape}, 类别分布: {np.bincount(y_raw)}")
        print(f"共有 {len(np.unique(groups_raw))} 个原始试次")

        # 生成参考信号（注意长度需匹配）
        Yf = generate_cca_references(
            TARGET_FREQS,
            srate=SAMPLE_RATE,
            T=L / SAMPLE_RATE,
            n_harmonics=5
        )
        print(f"参考信号 Yf shape: {Yf.shape}")

        # 选择模型：这里固定使用 FBTDCA（你之前的最优模型）
        best_name = "FBTDCA"
        model_class = FBTDCA
        kwargs = {
            'filterbank': filterbank,
            'padding_len': 0,
            'n_components': N_COMPONENTS_TDCA,
            'filterweights': filterweights
        }

        # 5折交叉验证
        mean_acc, std_acc = evaluate_model_kfold(
            model_class, X_raw, y_raw, groups_raw,
            Yf=Yf, n_splits=5, **kwargs
        )
        print(f"{best_name} 5折交叉验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")

        # 用全部数据增强后训练最终模型
        print(f"使用全部数据增强后训练最终模型...")
        X_all_aug, y_all_aug = augment_ssvep_data(
            X_raw, y_raw,
            num_aug=AUG_NUM,
            drop_prob=AUG_DROP_PROB,
            noise_ratio=AUG_NOISE_RATIO,
            scale_range=AUG_SCALE_RANGE,
            atten_range=AUG_ATTEN_RANGE,
            random_seed=2025
        )
        final_model = model_class(**kwargs)
        final_model.fit(X_all_aug, y_all_aug, Yf=Yf)

        # 保存模型
        model_path = model_names[L]
        joblib.dump(final_model, model_path)
        print(f"模型已保存为 {model_path}")

    print("\n所有模型训练完成！")

if __name__ == "__main__":
    main()