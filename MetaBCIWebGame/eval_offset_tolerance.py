# eval_offset_tolerance.py
import os
import re
import numpy as np
import joblib
from collections import defaultdict
from sklearn.metrics import accuracy_score

# 加载模型
model = joblib.load("self_ssvep_model.pkl")
DATA_ROOT = r"D:\pycharm\PyCharm 2026.1\my-projects\MetaBCI\data_self_multi_offset"
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]

def load_by_offset(root, offset_str):
    X_list, y_list = [], []
    for label in range(4):
        dir_path = os.path.join(root, str(label+1))
        for fname in os.listdir(dir_path):
            if not fname.endswith('.npy') or offset_str not in fname:
                continue
            data = np.load(os.path.join(dir_path, fname))
            data = data[OCCIPITAL_INDICES, :]
            data = data - np.mean(data, axis=1, keepdims=True)
            X_list.append(data[np.newaxis, ...])
            y_list.append(label)
    return np.vstack(X_list), np.array(y_list)

# 测试每个偏移
offsets = ['offset000', 'offset025', 'offset050', 'offset075', 'offset100', 'offset125']
for off in offsets:
    X, y = load_by_offset(DATA_ROOT, off)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"{off}: {acc*100:.2f}%")