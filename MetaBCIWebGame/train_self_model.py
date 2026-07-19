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
from config import BASE_DIR
DATA_ROOT = os.path.join(BASE_DIR, "data_self_multi_offset")

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
AUG_NUM = 5                     # 每个原始试次生成的增强样本数（提升至5对抗分布偏移）
AUG_DROP_PROB = 0.6             # 触发通道衰减的概率
AUG_NOISE_RATIO = 0.015         # 噪声幅值为该通道标准差的 1.5%（比实验环境更激进）
AUG_SCALE_RANGE = (0.7, 1.3)    # 整体幅值缩放范围（加宽模拟游戏中的阻抗漂移）
AUG_ATTEN_RANGE = (0.3, 0.8)    # 单通道衰减保留比例
AUG_FREQ_JITTER = 0.03          # 频率抖动：时间轴最大拉伸/压缩比例
AUG_BLINK_PROB = 0.3            # 注入眨眼伪迹的概率
AUG_BLINK_AMP = 5.0             # 眨眼伪迹最大幅值（μV量级，相对归一化数据）
AUG_TEMPORAL_SHIFT = 25         # 随机窗口偏移量（采样点）


# ============================================================
#  数据增强（实验→游戏分布偏移对抗版）
#
#  在原有三个增强（幅值缩放、通道衰减、高斯噪声）基础上新增：
#  ④ 频率抖动 — 模拟游戏中注意力波动导致的 SSVEP 频率漂移
#  ⑤ 眨眼伪迹 — 模拟游戏中更频繁的眼动/眨眼
#  ⑥ 时间偏移 — 模拟游戏中自定节奏的试次对齐误差
# ============================================================
def augment_ssvep_data(X, y, num_aug=AUG_NUM, drop_prob=AUG_DROP_PROB,
                       noise_ratio=AUG_NOISE_RATIO,
                       scale_range=AUG_SCALE_RANGE, atten_range=AUG_ATTEN_RANGE,
                       random_seed=None):
    """
    针对 SSVEP 固定窗口数据的增强策略（分布偏移对抗版）
    - ① 整体幅值缩放（模拟阻抗/皮肤导电率变化）
    - ② 随机单通道衰减（模拟个别电极接触变差）
    - ③ 按通道自适应高斯噪声（模拟系统热噪声）
    - ④ 频率抖动 / 时间轴非线性拉伸（模拟注意力波动→SSVEP频率漂移）
    - ⑤ 眨眼伪迹注入（游戏中更频繁的眼动/眨眼）
    - ⑥ 随机窗口偏移（游戏中自定节奏的试次时间对齐误差）
    """
    rng = np.random.RandomState(random_seed)
    X_aug, y_aug = [], []
    n_trials, n_chans, n_samples = X.shape

    # 眨眼模板：高斯波形模拟眨眼（仅影响额区通道 Fp1/Fp2，即索引0,1）
    blink_template = _make_blink_template(n_samples, peak_amp=AUG_BLINK_AMP)

    for i in range(n_trials):
        original = X[i].astype(np.float64)  # (n_chans, n_samples)

        # --- 1. 保留原始样本 ---
        X_aug.append(original.copy())
        y_aug.append(y[i])

        # --- 2. 生成增强样本 ---
        for aug_idx in range(num_aug):
            aug = original.copy()

            # ① 整体幅值缩放（更宽范围模拟游戏中的大幅度阻抗漂移）
            scale = rng.uniform(*scale_range)
            aug *= scale

            # ② 多通道衰减（游戏中可能多个电极松动）
            n_drop = rng.randint(1, max(2, n_chans // 3))
            drop_chs = rng.choice(n_chans, size=n_drop, replace=False)
            for ch in drop_chs:
                atten = rng.uniform(*atten_range)
                aug[ch, :] *= atten

            # ③ 按通道自适应高斯噪声
            sigma = noise_ratio * np.std(aug, axis=1, keepdims=True)
            aug += rng.normal(0, sigma, aug.shape)

            # ④ 频率抖动：对时间轴做非线性拉伸/压缩
            if rng.rand() < 0.6:
                aug = _apply_frequency_jitter(aug, rng, jitter=AUG_FREQ_JITTER)

            # ⑤ 眨眼伪迹注入：仅在额区通道添加
            if rng.rand() < AUG_BLINK_PROB:
                blink_pos = rng.randint(0, n_samples - len(blink_template))
                aug[0, blink_pos:blink_pos + len(blink_template)] += blink_template
                # 有时两眼同步眨眼
                if rng.rand() < 0.7:
                    aug[1, blink_pos:blink_pos + len(blink_template)] += \
                        blink_template * rng.uniform(0.7, 1.0)

            # ⑥ 随机窗口偏移：滚动数据模拟时间对齐误差
            shift = rng.randint(-AUG_TEMPORAL_SHIFT, AUG_TEMPORAL_SHIFT + 1)
            if shift != 0:
                aug = np.roll(aug, shift, axis=-1)
                if shift > 0:
                    aug[:, :shift] = aug[:, shift:shift + 1]  # 边界填充
                else:
                    aug[:, shift:] = aug[:, shift - 1:shift]

            X_aug.append(aug)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


def _make_blink_template(n_samples, peak_amp=5.0):
    """生成一个高斯波形的眨眼伪迹模板（宽度约200ms @ 250Hz = 50点）。"""
    t = np.arange(n_samples)
    # 随机位置的高斯脉冲，sigma ≈ 25 点 (100 ms)
    center = n_samples // 4  # 模板中心在前 1/4 处，后续随机移位
    sigma = 25.0
    template = peak_amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return template.astype(np.float64)


def _apply_frequency_jitter(aug, rng, jitter=0.03):
    """对单试次数据做时间轴非线性拉伸/压缩，模拟 SSVEP 频率微小漂移。

    使用正弦调制的时间重采样：t' = t + α * sin(2π * t * f_mod)
    其中 α 控制抖动幅度，f_mod 为调制频率（低频，模拟慢漂移）。
    """
    n_chans, n_samples = aug.shape
    t = np.arange(n_samples, dtype=np.float64)
    # 随机调制参数
    alpha = rng.uniform(0, jitter) * n_samples   # 最大位移（采样点）
    f_mod = rng.uniform(0.5, 3.0) / n_samples    # 调制频率（慢变）
    # 新时间坐标
    t_new = t + alpha * np.sin(2.0 * np.pi * f_mod * t + rng.uniform(0, 2 * np.pi))
    t_new = np.clip(t_new, 0, n_samples - 1)
    # 逐通道插值
    result = np.zeros_like(aug)
    for ch in range(n_chans):
        result[ch] = np.interp(t_new, t, aug[ch])
    return result


# ============================================================
#  数据加载（修改2：返回 groups）
# ============================================================
def load_data_from_dirs(root, use_occipital=True, window_samples=None,
                        offset_only=True):
    """
    加载原始数据（不做增强）。
    window_samples: 截取前 N 个采样点，None=全取
    offset_only: 仅加载 offset000 文件
    返回: X (n_trials, n_channels, samples), y (n_trials,), groups (n_trials,)
    """
    X_list, y_list, groups_list = [], [], []

    for label in range(4):
        dir_path = os.path.join(root, str(label + 1))
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
        for fname in files:
            if offset_only and "offset000" not in fname:
                continue

            data = np.load(os.path.join(dir_path, fname))
            if window_samples is not None:
                if data.shape[1] < window_samples:
                    continue
                data = data[:, :window_samples]
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