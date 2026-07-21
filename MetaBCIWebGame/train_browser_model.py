# train_browser_model.py
# -*- coding: utf-8 -*-
"""
用浏览器 Canvas 采集的数据训练 SSVEP 模型。
采集数据位于 data_self_browser/{最新时间戳}/ 目录。
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from metabci.brainda.algorithms.decomposition.tdca import FBTDCA
from metabci.brainda.algorithms.decomposition.cca import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "browser")
TARGET_FREQS = [8.25, 11.0, 13.75, 16.5]
WINDOW_SAMPLES = 500
SAMPLE_RATE = 250
N_COMPONENTS_TDCA = 3
N_COMPONENTS_TRCA = 1
ENSEMBLE = True
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]

# ---- 找到最新的采集目录 ----
browser_root = os.path.join(BASE_DIR, "data_self_browser")
if not os.path.isdir(browser_root):
    print(f"错误: 采集目录不存在 {browser_root}")
    sys.exit(1)

sessions = sorted(glob.glob(os.path.join(browser_root, "*")))
if not sessions:
    print(f"错误: 采集目录为空")
    sys.exit(1)

DATA_ROOT = sessions[-1]  # 最新一次采集
print(f"数据目录: {DATA_ROOT}")


def load_data(root):
    """加载浏览器采集数据（每试次一个.npy文件，label目录结构）。"""
    X_list, y_list, groups_list = [], [], []
    for label in range(4):
        dir_path = os.path.join(root, str(label + 1))
        if not os.path.isdir(dir_path):
            continue
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])
        for fname in files:
            data = np.load(os.path.join(dir_path, fname))
            if data.shape[1] < WINDOW_SAMPLES:
                continue
            data = data[:, :WINDOW_SAMPLES]
            # 15通道格式(14EEG+Trigger): 只用前14通道训练
            if data.shape[0] == 15:
                data = data[:14, :]
            X_list.append(data)
            y_list.append(label)
            groups_list.append(len(groups_list))  # 每个文件作为一个group
    if not X_list:
        raise RuntimeError("未加载到任何数据")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    groups = np.array(groups_list)
    X = X[:, OCCIPITAL_INDICES, :]
    X = X - np.mean(X, axis=2, keepdims=True)
    X = X + 1e-10 * np.random.randn(*X.shape)
    return X, y, groups


# ---- 加载数据 ----
X, y, groups = load_data(DATA_ROOT)
print(f"数据: {X.shape}, 类别: {np.bincount(y)}")

# ---- 滤波器组 ----
passbands = [[6, 90], [14, 90], [22, 90], [30, 90], [38, 90]]
stopbands = [[4, 100], [10, 100], [16, 100], [24, 100], [32, 100]]
filterbank = generate_filterbank(passbands, stopbands, SAMPLE_RATE)
fweights = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(len(passbands))])
Yf = generate_cca_references(TARGET_FREQS, srate=SAMPLE_RATE, T=WINDOW_SAMPLES / SAMPLE_RATE, n_harmonics=5)

# ---- 5折交叉验证 ----
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
accs_tdca, accs_trca = [], []

for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    tdca = FBTDCA(filterbank=filterbank, padding_len=0, n_components=N_COMPONENTS_TDCA, filterweights=fweights)
    tdca.fit(X_tr, y_tr, Yf=Yf)
    accs_tdca.append(accuracy_score(y_te, tdca.predict(X_te)))

    trca = FBTRCA(filterbank=filterbank, n_components=N_COMPONENTS_TRCA, ensemble=ENSEMBLE, filterweights=fweights)
    trca.fit(X_tr, y_tr)
    accs_trca.append(accuracy_score(y_te, trca.predict(X_te)))

print(f"\n5折交叉验证: FBTDCA={np.mean(accs_tdca)*100:.1f}% FBTRCA={np.mean(accs_trca)*100:.1f}%")

# ---- 全部数据训练最优模型 ----
best_name = 'FBTDCA' if np.mean(accs_tdca) > np.mean(accs_trca) else 'FBTRCA'
print(f"最优: {best_name}")

if best_name == 'FBTDCA':
    final = FBTDCA(filterbank=filterbank, padding_len=0, n_components=N_COMPONENTS_TDCA, filterweights=fweights)
    final.fit(X, y, Yf=Yf)
else:
    final = FBTRCA(filterbank=filterbank, n_components=N_COMPONENTS_TRCA, ensemble=ENSEMBLE, filterweights=fweights)
    final.fit(X, y)

# ---- 保存 ----
joblib.dump(final, os.path.join(MODEL_DIR, "self_ssvep_model_browser.pkl"))
print("模型已保存: self_ssvep_model_browser.pkl")

# ---- 也训练Growing Window模型 ----
print("\n训练 Growing Window 模型...")
for L in [125, 250, 375, 500]:
    events = []
    for label in range(4):
        d = os.path.join(DATA_ROOT, str(label + 1))
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.npy'):
                trial = np.load(os.path.join(d, fn))
                if trial.shape[1] >= L:
                    events.append((trial, label))
    if not events:
        print(f"  L={L}: 无数据")
        continue
    X_L = np.stack([e[0][OCCIPITAL_INDICES, :L] for e in events])
    y_L = np.array([e[1] for e in events])
    X_L = X_L - np.mean(X_L, axis=2, keepdims=True)
    X_L = X_L + 1e-10 * np.random.randn(*X_L.shape)
    groups_L = np.arange(len(y_L))

    sgkf_L = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    accs_L = []
    Yf_L = generate_cca_references(TARGET_FREQS, srate=SAMPLE_RATE, T=L / SAMPLE_RATE, n_harmonics=5)
    for tr, te in sgkf_L.split(X_L, y_L, groups_L):
        m = FBTDCA(filterbank=filterbank, padding_len=0, n_components=N_COMPONENTS_TDCA, filterweights=fweights)
        m.fit(X_L[tr], y_L[tr], Yf=Yf_L)
        accs_L.append(accuracy_score(y_L[te], m.predict(X_L[te])))
    print(f"  L={L}: CV={np.mean(accs_L)*100:.1f}%")

    final_L = FBTDCA(filterbank=filterbank, padding_len=0, n_components=N_COMPONENTS_TDCA, filterweights=fweights)
    final_L.fit(X_L, y_L, Yf=Yf_L)
    fname = {125: 'model_125_browser.pkl', 250: 'model_250_browser.pkl',
             375: 'model_375_browser.pkl', 500: 'self_ssvep_model_browser.pkl'}[L]
    joblib.dump(final_L, os.path.join(MODEL_DIR, fname))

print("\n全部模型已保存。使用方式:")
print("  修改 config.py GW_MODEL_PATHS 指向上面的 *_browser.pkl 文件")
