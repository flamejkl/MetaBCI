# offline_train_public.py
"""
基于 Wang2016 公开数据集的 SSVEP 离线训练与评估
- 通道数：14 导（可修改为 8 导枕顶区）
- 窗口长度：2 秒（精确 500 个采样点，无延迟）
- 评估方式：留一被试 (LOSO) 在前30人内进行
- 支持 FBTDCA 算法（泛化性优于 FBTRCA）
- 最终模型保存为 best_ssvep_model_public.pkl
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import joblib
import mne
from mne.io import BaseRaw

from metabci.brainda.datasets.tsinghua import Wang2016
from metabci.brainda.algorithms.decomposition.cca import FBTRCA
from metabci.brainda.algorithms.decomposition.tdca import FBTDCA
from fbcca_eigh import get_default_filterbank

# ========== 参数配置 ==========
TARGET_FREQS = [8.0, 10.0, 12.0, 15.0]          # 目标频率（顺序对应上、下、左、右）
SUBJECTS = list(range(1, 31))                   # 前30名被试用于训练和验证
WINDOW_LEN_SEC = 2.0                            # 窗口长度（秒），严格等于 2 秒
DELAY = 0.0                                     # 刺激开始后的延迟（秒），改为 0 取消延迟
SAMPLE_RATE = 250                               # Wang2016 采样率
DATA_ROOT = r"E:\XYLData"                       # 数据存储根目录

# 目标通道列表（14通道，可根据需要修改为8导枕顶区）
SELECTED_CHANNELS = [
    'FP1', 'FP2', 'O1', 'O2', 'OZ',
    'PO3', 'PO4', 'PO5', 'PO6', 'POZ',
    'P3', 'P4', 'P7', 'P8'
]
# 可选：8导枕顶区（取消注释即可使用）
# SELECTED_CHANNELS = [
#     'O1', 'O2', 'OZ',
#     'PO3', 'PO4', 'PO5', 'PO6', 'POZ'
# ]

# 算法选择：'FBTRCA' 或 'FBTDCA'
ALGORITHM = 'FBTDCA'   # 改为 'FBTRCA' 可回退

# FBTDCA 参数（仅在 ALGORITHM == 'FBTDCA' 时生效）
L_DELAY_SAMPLES = 0          # 延迟样本数（250Hz 下 0 表示无延迟）
N_COMPONENTS_TDCA = 8        # 空间滤波器个数

# FBTRCA 参数
N_COMPONENTS_TRCA = 2        # 空间滤波器个数
ENSEMBLE = True              # 是否使用集成

PADDING_LEN = L_DELAY_SAMPLES   # 填充点数，与延迟样本数一致（现在为 0）

# ---------- 辅助函数：递归提取所有 Raw 对象 ----------
def _extract_all_raws(obj):
    raws = []
    if isinstance(obj, BaseRaw):
        raws.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            raws.extend(_extract_all_raws(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            raws.extend(_extract_all_raws(item))
    return raws

# ---------- 加载单个被试的数据 ----------
def load_wang2016_subject(subject, selected_chs, delay=DELAY, window_sec=WINDOW_LEN_SEC):
    """
    加载单个被试的数据，支持延迟，并精确控制窗口长度。
    返回 X (n_trials, n_channels, n_samples), y (labels)
    """
    dataset = Wang2016()
    dataset.data_path(subject=subject, path=DATA_ROOT, update_path=True)
    raw_dict = dataset.get_data(subjects=[subject])
    raws = _extract_all_raws(raw_dict)
    if not raws:
        raise ValueError(f"被试 {subject} 没有找到任何 Raw 对象")
    print(f"  发现 {len(raws)} 个 Raw 对象")

    X_list, y_list = [], []
    for raw in raws:
        events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
        if len(events) == 0:
            continue
        # 选择通道
        avail_chs = [ch for ch in selected_chs if ch in raw.ch_names]
        if len(avail_chs) != len(selected_chs):
            missing = set(selected_chs) - set(avail_chs)
            print(f"    警告：缺少通道 {missing}，跳过该 Raw")
            continue
        raw_eeg = raw.copy().pick_channels(avail_chs, verbose=False)

        # 精确的 epoch 边界（包含延迟，不包含终点）
        tmin = delay
        # 保证采样点数 = (tmax - tmin) * srate + 1 = window_sec * srate
        # 由于 tmin=0，tmax = window_sec - 1/srate
        tmax = delay + window_sec - 1.0 / SAMPLE_RATE
        epochs = mne.Epochs(raw_eeg, events, event_id=None, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, verbose=False)
        X_list.append(epochs.get_data())
        y_list.append(epochs.events[:, -1])

    if not X_list:
        raise ValueError(f"被试 {subject} 没有生成任何 epochs")
    X = np.concatenate(X_list, axis=0)
    y_raw = np.concatenate(y_list, axis=0)

    # 将原始事件ID映射为目标频率标签 (0~3)
    dataset_tmp = Wang2016()
    freq_list = dataset_tmp._FREQS
    target_indices = {freq: idx for idx, freq in enumerate(TARGET_FREQS)}
    y_mapped = []
    mask = []
    for i, ev in enumerate(y_raw):
        if 1 <= ev <= len(freq_list):
            freq = freq_list[ev - 1]
            if freq in target_indices:
                y_mapped.append(target_indices[freq])
                mask.append(i)
    if not y_mapped:
        raise ValueError(f"被试 {subject} 无目标频率数据")
    X = X[mask]
    y = np.array(y_mapped)
    return X, y

# ---------- 主程序 ----------
print("正在加载 Wang2016 数据集（前30名被试）...")
all_X, all_y, all_subjects = [], [], []
for sub in SUBJECTS:
    print(f"处理被试 {sub}...")
    try:
        X_sub, y_sub = load_wang2016_subject(sub, SELECTED_CHANNELS)
        if len(X_sub) == 0:
            continue
        all_X.append(X_sub)
        all_y.append(y_sub)
        all_subjects.extend([sub] * len(X_sub))
        print(f"  加载成功: {X_sub.shape}, 标签分布: {np.bincount(y_sub)}")
    except Exception as e:
        print(f"  被试 {sub} 加载失败: {e}")

if not all_X:
    raise RuntimeError("未加载到任何有效数据")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
subjects_arr = np.array(all_subjects)
print(f"总数据: {X.shape}, 标签分布: {np.bincount(y)}")
print(f"被试分布: {np.unique(subjects_arr, return_counts=True)}")

# 生成滤波器组和权重（与训练保持一致）
filterbank, filterweights = get_default_filterbank(SAMPLE_RATE)
filterweights = np.array(filterweights)

# 生成所有频率的参考信号 Yf（FBTDCA 需要）
if ALGORITHM == 'FBTDCA':
    from metabci.brainda.algorithms.decomposition.base import generate_cca_references
    dataset_full = Wang2016()
    events = list(dataset_full.events.keys())
    all_freqs = [dataset_full.get_freq(event) for event in events]
    n_samples = int(WINDOW_LEN_SEC * SAMPLE_RATE)  # 此处计算为 500，用于参考信号生成
    Yf = generate_cca_references(all_freqs, srate=SAMPLE_RATE, T=WINDOW_LEN_SEC,
                                 n_harmonics=5)   # 谐波次数可调整
    print(f"已生成 {len(all_freqs)} 个频率的参考信号，形状: {Yf.shape}")
else:
    Yf = None

# ==================== 留一被试评估（LOSO）====================
print("\n========== 留一被试评估 (LOSO) 前30人 ==========")
print(f"算法: {ALGORITHM}, 窗口: {WINDOW_LEN_SEC}s, 延迟: {DELAY}s")
unique_subs = np.unique(subjects_arr)
loso_acc = []

for test_sub in unique_subs:
    test_mask = subjects_arr == test_sub
    train_mask = ~test_mask
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"测试被试 {test_sub}...")
    if ALGORITHM == 'FBTRCA':
        model = FBTRCA(filterbank=filterbank, n_components=N_COMPONENTS_TRCA,
                       ensemble=ENSEMBLE, filterweights=filterweights)
    elif ALGORITHM == 'FBTDCA':
        model = FBTDCA(filterbank=filterbank, padding_len=L_DELAY_SAMPLES,
                       n_components=N_COMPONENTS_TDCA,
                       filterweights=filterweights)
    else:
        raise ValueError(f"未知算法: {ALGORITHM}")

    if ALGORITHM == 'FBTRCA':
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, Yf=Yf)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    loso_acc.append(acc)
    print(f"  准确率: {acc:.4f}")

print(f"\nLOSO 平均准确率: {np.mean(loso_acc):.4f} ± {np.std(loso_acc):.4f}")

# ==================== 最终模型保存 ====================
print(f"\n使用全部前30人数据训练最终模型 ({ALGORITHM})...")
if ALGORITHM == 'FBTRCA':
    final_model = FBTRCA(filterbank=filterbank, n_components=N_COMPONENTS_TRCA,
                         ensemble=ENSEMBLE, filterweights=filterweights)
    final_model.fit(X, y)
else:
    final_model = FBTDCA(filterbank=filterbank, padding_len=L_DELAY_SAMPLES,
                         n_components=N_COMPONENTS_TDCA,
                         filterweights=filterweights)
    final_model.fit(X, y, Yf=Yf)

joblib.dump(final_model, "best_ssvep_model_public.pkl")
print("最终模型已保存为 best_ssvep_model_public.pkl")