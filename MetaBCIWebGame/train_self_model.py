# train_self_model.py
# -*- coding: utf-8 -*-
"""
从 data_self 目录加载数据，训练 FBTDCA 和 FBTRCA，
使用 5 折交叉验证评估（训练集增强，测试集不增强），选择最优算法保存模型。

数据增强策略（适配 500 点固定窗口）：
1. 整体幅值缩放（模拟阻抗/皮肤导电率变化）
2. 单通道衰减（模拟电极接触变差）
3. 按通道自适应高斯噪声（模拟系统热噪声）

修改说明：使用 StratifiedGroupKFold 防止同一 trial 的不同偏移窗口被拆分到不同 Fold。
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
#  配置参数（可根据实际效果微调）
# ============================================================
# 请改为你新生成的多偏移数据集路径
DATA_ROOT = r"D:\pycharm\PyCharm 2026.1\my-projects\MetaBCI\data_self_multi_offset"

TARGET_FREQS = [8.25, 11.0, 13.75, 16.5]
WINDOW_SAMPLES = 500
SAMPLE_RATE = 250

# 模型超参数
N_COMPONENTS_TDCA = 3
N_COMPONENTS_TRCA = 1
ENSEMBLE = True

# 枕区通道索引（根据你的 14 导布局调整，此处索引 2~9 对应 O1, O2, Oz, PO3...）
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]

# ========== 数据增强参数 ==========
AUG_NUM = 3                     # 每个原始试次生成的增强样本数
AUG_DROP_PROB = 0.5             # 触发通道衰减的概率
AUG_NOISE_RATIO = 0.01          # 噪声幅值为该通道标准差的 1%
AUG_SCALE_RANGE = (0.9, 1.1)    # 整体幅值缩放范围
AUG_ATTEN_RANGE = (0.4, 0.7)    # 单通道衰减保留比例


# ============================================================
#  数据增强：幅值缩放 + 通道衰减 + 按通道自适应噪声
# ============================================================
def augment_ssvep_data(X, y, num_aug=3, drop_prob=0.5, noise_ratio=0.01,
                       scale_range=(0.9, 1.1), atten_range=(0.4, 0.7),
                       random_seed=None):
    """
    针对 SSVEP 固定窗口数据的增强策略
    - 整体幅值缩放（模拟皮肤阻抗变化）
    - 随机单通道衰减（模拟个别电极接触变差）
    - 按通道标准差的自适应高斯噪声（模拟系统热噪声）
    """
    rng = np.random.RandomState(random_seed)
    X_aug, y_aug = [], []
    n_trials, n_chans, n_samples = X.shape

    for i in range(n_trials):
        original = X[i]  # (n_chans, n_samples)

        # --- 1. 保留原始样本 ---
        X_aug.append(original)
        y_aug.append(y[i])

        # --- 2. 生成增强样本 ---
        for _ in range(num_aug):
            aug = original.copy()

            # ① 整体幅值缩放
            scale = rng.uniform(*scale_range)
            aug *= scale

            # ② 通道衰减
            if rng.rand() < drop_prob:
                ch = rng.randint(0, n_chans)
                atten = rng.uniform(*atten_range)
                aug[ch, :] *= atten

            # ③ 按通道自适应高斯噪声
            sigma = noise_ratio * np.std(aug, axis=1, keepdims=True)
            aug = aug + rng.normal(0, sigma, aug.shape)

            X_aug.append(aug)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


# ============================================================
#  数据加载（修改2：返回 groups）
# ============================================================
def load_data_from_dirs(root, use_occipital=True):
    """
    加载原始数据（不做增强）。
    返回: X (n_trials, n_channels, 500), y (n_trials,), groups (n_trials,)
    """
    X_list, y_list = [], []
    groups_list = []

    for label in range(4):
        dir_path = os.path.join(root, str(label + 1))
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
        for fname in files:
            # ========== 新增：只加载偏移 0 的文件 ==========
            if "offset000" not in fname:
                continue
            # =============================================

            data = np.load(os.path.join(dir_path, fname))
            X_list.append(data)
            y_list.append(label)

            match = re.search(r'hw_trial_(\d+)', fname)
            if match:
                group_id = int(match.group(1))
            else:
                group_id = len(groups_list)
            groups_list.append(group_id)

    if not X_list:
        raise RuntimeError("未加载到任何数据！")

    X = np.stack(X_list, axis=0)  # (n, 14, 500)
    y = np.array(y_list)
    groups = np.array(groups_list)

    # 选择枕区通道
    if use_occipital:
        X = X[:, OCCIPITAL_INDICES, :]
        print(f"使用枕区 {len(OCCIPITAL_INDICES)} 个通道，新形状: {X.shape}")

    # 去均值 + 微小平稳噪声（数值稳定）
    X = X - np.mean(X, axis=2, keepdims=True)
    X = X + 1e-10 * np.random.randn(*X.shape)

    return X, y, groups


# ============================================================
#  交叉验证评估（修改3：使用 StratifiedGroupKFold）
# ============================================================
def evaluate_model_kfold(model_class, X, y, groups, Yf=None, n_splits=5, **kwargs):
    """
    分层分组 K 折交叉验证（StratifiedGroupKFold）。
    保证同一原始试次（trial_id）的所有偏移窗口不被拆分到不同 Fold。
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    fold = 0

    for train_idx, test_idx in sgkf.split(X, y, groups):  # 传入 groups
        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # ---- 训练集增强 ----
        X_train_aug, y_train_aug = augment_ssvep_data(
            X_train_raw, y_train_raw,
            num_aug=AUG_NUM,
            drop_prob=AUG_DROP_PROB,
            noise_ratio=AUG_NOISE_RATIO,
            scale_range=AUG_SCALE_RANGE,
            atten_range=AUG_ATTEN_RANGE,
            random_seed=42 + fold  # 每折不同种子
        )

        # ---- 模型训练 ----
        if Yf is not None:
            model = model_class(**kwargs)
            model.fit(X_train_aug, y_train_aug, Yf=Yf)
        else:
            model = model_class(**kwargs)
            model.fit(X_train_aug, y_train_aug)

        # ---- 测试集预测（原始数据，不增强） ----
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))

        fold += 1

    return np.mean(accs), np.std(accs)


# ============================================================
#  主程序（修改4：接收并使用 groups）
# ============================================================
def main():
    print("===== 训练并评估 SSVEP 模型（多策略增强：缩放+衰减+按通道噪声）=====")
    print("使用 StratifiedGroupKFold 防止同一 trial 偏移窗口泄漏")

    # 1. 加载原始数据（不增强）
    X_raw, y_raw, groups_raw = load_data_from_dirs(DATA_ROOT, use_occipital=True)
    print(f"原始数据形状: {X_raw.shape}，类别分布: {np.bincount(y_raw)}")
    print(f"共有 {len(np.unique(groups_raw))} 个原始试次（每个试次含多个偏移窗口）")
    print(f"当前试次长度为 {X_raw.shape[2]} 点（固定窗口），增强策略已适配该长度。")

    # 2. 生成滤波器组与参考信号
    filterbank, filterweights = get_default_filterbank(SAMPLE_RATE)
    filterweights = np.array(filterweights)

    from metabci.brainda.algorithms.decomposition.base import generate_cca_references
    Yf = generate_cca_references(
        TARGET_FREQS,
        srate=SAMPLE_RATE,
        T=WINDOW_SAMPLES / SAMPLE_RATE,
        n_harmonics=5
    )
    print(f"参考信号 Yf shape: {Yf.shape}")

    # 3. 交叉验证比较两种模型
    print("\n===== 5 折分层分组交叉验证（训练集增强，测试集原始）=====")
    models = {
        'FBTRCA': (
            FBTRCA,
            {
                'filterbank': filterbank,
                'n_components': N_COMPONENTS_TRCA,
                'ensemble': ENSEMBLE,
                'filterweights': filterweights
            }
        ),
        'FBTDCA': (
            FBTDCA,
            {
                'filterbank': filterbank,
                'padding_len': 0,
                'n_components': N_COMPONENTS_TDCA,
                'filterweights': filterweights
            }
        )
    }

    results = {}
    for name, (cls, kwargs) in models.items():
        print(f"\n评估 {name}...")
        if name == 'FBTDCA':
            mean_acc, std_acc = evaluate_model_kfold(
                cls, X_raw, y_raw, groups_raw, Yf=Yf, n_splits=5, **kwargs
            )
        else:
            mean_acc, std_acc = evaluate_model_kfold(
                cls, X_raw, y_raw, groups_raw, Yf=None, n_splits=5, **kwargs
            )
        results[name] = mean_acc
        print(f"{name} 5折交叉验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")

    # 4. 选择最优模型
    best_name = max(results, key=results.get)
    best_acc = results[best_name]
    print(f"\n最优模型: {best_name}, 交叉验证准确率: {best_acc:.4f}")

    # 5. 用全部原始数据增强后训练最终模型
    print(f"\n使用全部原始数据增强后训练最终模型 ({best_name})...")
    X_all_aug, y_all_aug = augment_ssvep_data(
        X_raw, y_raw,
        num_aug=AUG_NUM,
        drop_prob=AUG_DROP_PROB,
        noise_ratio=AUG_NOISE_RATIO,
        scale_range=AUG_SCALE_RANGE,
        atten_range=AUG_ATTEN_RANGE,
        random_seed=2025  # 固定种子
    )
    print(f"增强后训练集大小: {X_all_aug.shape}")

    if best_name == 'FBTRCA':
        final_model = FBTRCA(
            filterbank=filterbank,
            n_components=N_COMPONENTS_TRCA,
            ensemble=ENSEMBLE,
            filterweights=filterweights
        )
        final_model.fit(X_all_aug, y_all_aug)
    else:
        final_model = FBTDCA(
            filterbank=filterbank,
            padding_len=0,
            n_components=N_COMPONENTS_TDCA,
            filterweights=filterweights
        )
        final_model.fit(X_all_aug, y_all_aug, Yf=Yf)

    model_path = "self_ssvep_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"最终模型已保存为 {model_path}")
    print("\n训练完成。注意：最终模型已在全部数据（含增强）上训练，请通过在线实测验证效果。")


if __name__ == "__main__":
    main()