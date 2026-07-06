# validate_threshold_robustness.py
# -*- coding: utf-8 -*-
"""
参数鲁棒性验证（修正 Growing Window，并收集 margin 分布）
"""
import os
import numpy as np
import joblib
from collections import deque
from sklearn.metrics import accuracy_score

# ============================================================
#  配置
# ============================================================
OPTIMAL_CONFIG = {
    'margin_threshold': 0.35,
    'max_threshold': 0.75,
    'consecutive_required': 1,
    'check_step': 25
}

from config import OFFLINE_DATA_ROOT
DATA_ROOT = OFFLINE_DATA_ROOT
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]
SAMPLE_RATE = 250
WINDOW_LENGTHS = [125, 250, 375, 500]
MODEL_PATHS = {
    125: "model_125.pkl",
    250: "model_250.pkl",
    375: "model_375.pkl",
    500: "self_ssvep_model.pkl"
}

# ============================================================
#  加载模型与数据
# ============================================================
print("加载模型...")
models = {}
for L, path in MODEL_PATHS.items():
    try:
        models[L] = joblib.load(path)
        print(f"✅ 加载 {path} (长度 {L} 点)")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        exit(1)

def load_all_trials(root):
    X_list, y_list = [], []
    for label in range(4):
        dir_path = os.path.join(root, str(label+1))
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy') and 'offset' not in f]
        for fname in files:
            data = np.load(os.path.join(dir_path, fname))
            data = data[OCCIPITAL_INDICES, :]
            X_list.append(data)
            y_list.append(label)
    return np.stack(X_list, axis=0), np.array(y_list)

print("\n加载数据...")
X, y_true = load_all_trials(DATA_ROOT)
print(f"加载 {X.shape[0]} 个试次")

# ============================================================
#  预处理与分数获取
# ============================================================
def preprocess(window):
    window = window - np.mean(window, axis=1, keepdims=True)
    from scipy.signal import butter, sosfilt
    fs = SAMPLE_RATE
    sos = butter(4, [15.5, 17.5], btype='bandpass', fs=fs, output='sos')
    filtered = sosfilt(sos, window, axis=-1)
    window = window + 0.5 * filtered
    return window

def get_scores(window, model):
    X_input = window[np.newaxis, ...]
    if hasattr(model, 'transform'):
        return model.transform(X_input)[0]
    else:
        label = model.predict(X_input)[0]
        scores = np.zeros(4)
        if 0 <= label < 4:
            scores[label] = 1.0
        else:
            scores = np.ones(4) * 0.25
        return scores

# ============================================================
#  全局变量：收集所有检测点信息
# ============================================================
all_detections = []  # 每个元素为 {'time':, 'margin':, 'max':}

# ============================================================
#  模拟决策（修正 Growing Window）
# ============================================================
def simulate_decision(trial_data, true_label, config, trial_idx=0):
    global all_detections  # 使用全局变量收集

    # 自动修正维度方向
    if trial_data.shape[0] > trial_data.shape[1]:
        print(f"⚠️ [Trial {trial_idx}] 检测到维度方向错误 ({trial_data.shape})，自动转置为 (channels, time)")
        trial_data = trial_data.T
        print(f"   修正后形状: {trial_data.shape}")

    step = config.get('check_step', 25)
    margin_th = config['margin_threshold']
    max_th = config['max_threshold']
    cons_req = config['consecutive_required']

    history = deque(maxlen=cons_req)
    last_decision = None

    if trial_idx < 5:
        print(f"[Trial {trial_idx}] 开始模拟，真实标签={true_label}")

    # ===== 核心修正：直接使用 pos 作为窗口长度 =====
    for pos in range(step, trial_data.shape[1] + 1, step):
        L = pos  # 当前累积的样本点数

        if trial_idx < 5:
            print(f"[Trial {trial_idx}] pos={pos}, L={L}")

        if L < min(WINDOW_LENGTHS):
            if trial_idx < 5:
                print(f"  [跳过] L={L} < {min(WINDOW_LENGTHS)}")
            continue

        # 选择 <= L 的最大模型
        model_len = None
        for wl in sorted(WINDOW_LENGTHS):
            if wl <= L:
                model_len = wl
            else:
                break
        if model_len is None:
            if trial_idx < 5:
                print(f"  [跳过] model_len 为 None")
            continue

        if trial_idx < 5:
            print(f"  model_len={model_len}")

        # ===== Growing Window 核心：取前 L 个点 =====
        window_full = trial_data[:, :L]  # (channels, L)
        # 取该模型需要的长度
        window_trim = window_full[:, -model_len:] if L >= model_len else window_full

        window_trim = preprocess(window_trim)
        scores = get_scores(window_trim, models[model_len])

        top2 = np.partition(scores, -2)[-2:]
        margin = top2.max() - top2.min()
        max_score = np.max(scores)
        decision = np.argmax(scores)

        # ===== 收集检测点信息（全局） =====
        all_detections.append({
            'trial': trial_idx,
            'time': pos / SAMPLE_RATE,
            'margin': margin,
            'max': max_score
        })

        if trial_idx < 5:
            print(f"  scores={scores}, max={max_score:.4f}, margin={margin:.4f}")

        # 停止条件判断
        if margin > margin_th and max_score > max_th:
            history.append(decision)
            if len(history) == cons_req and len(set(history)) == 1:
                if trial_idx < 5:
                    print(f"✅ 提前停止于 {pos/SAMPLE_RATE:.2f}s")
                return pos / SAMPLE_RATE, decision, max_score
        else:
            history.clear()

    # 强制 2.0s 输出
    window_final = trial_data[:, :500] if trial_data.shape[1] >= 500 else trial_data
    window_final = preprocess(window_final)
    scores_final = get_scores(window_final, models[500])
    if trial_idx < 5:
        print(f"⏰ 未提前停止，强制 2.0s 输出")
    return 2.0, np.argmax(scores_final), np.max(scores_final)

# ============================================================
#  主验证循环
# ============================================================
print("\n" + "="*60)
print("🔍 参数鲁棒性验证（修正 Growing Window）")
print(f"当前阈值: margin={OPTIMAL_CONFIG['margin_threshold']}, "
      f"max={OPTIMAL_CONFIG['max_threshold']}, "
      f"连续={OPTIMAL_CONFIG['consecutive_required']}次")
print("-"*60)

# 清空全局检测列表
all_detections = []

results = []
for i in range(X.shape[0]):
    stop_time, decision, conf = simulate_decision(X[i], y_true[i], OPTIMAL_CONFIG, trial_idx=i)
    results.append({
        'true': y_true[i],
        'pred': decision,
        'time': stop_time,
        'conf': conf,
        'match': decision == y_true[i]
    })

# ============================================================
#  统计指标（原有）
# ============================================================
total = len(results)
accuracy = np.mean([r['match'] for r in results])
avg_time = np.mean([r['time'] for r in results])
std_time = np.std([r['time'] for r in results])
early_stop_rate = np.mean([r['time'] < 2.0 for r in results])

print("\n📊 总体性能:")
print(f"  准确率: {accuracy*100:.2f}%")
print(f"  平均决策时间: {avg_time*1000:.0f} ± {std_time*1000:.0f} ms")
print(f"  提前停止率: {early_stop_rate*100:.1f}%")

# ============================================================
#  新增：margin 分布统计（关键补充）
# ============================================================
print("\n📊 margin 分布（按时间点）:")
time_points = [0.5, 1.0, 1.5, 2.0]
for tp in time_points:
    # 由于浮点数精度，使用近似比较
    margins = [d['margin'] for d in all_detections if abs(d['time'] - tp) < 0.01]
    if margins:
        p50 = np.percentile(margins, 50)
        p70 = np.percentile(margins, 70)
        p90 = np.percentile(margins, 90)
        print(f"  {tp:.1f}s: 50%={p50:.3f}, 70%={p70:.3f}, 90%={p90:.3f}")
    else:
        print(f"  {tp:.1f}s: 无数据")

print("\n📊 max_score 分布（按时间点）:")
for tp in time_points:
    max_vals = [d['max'] for d in all_detections if abs(d['time'] - tp) < 0.01]
    if max_vals:
        p50 = np.percentile(max_vals, 50)
        p70 = np.percentile(max_vals, 70)
        p90 = np.percentile(max_vals, 90)
        print(f"  {tp:.1f}s: 50%={p50:.3f}, 70%={p70:.3f}, 90%={p90:.3f}")
    else:
        print(f"  {tp:.1f}s: 无数据")

# ============================================================
#  最终判断
# ============================================================
if accuracy >= 0.98 and early_stop_rate > 0.5:
    print("\n✅ 参数鲁棒性优秀，可以安全部署到在线系统。")
else:
    print("\n⚠️ 建议根据打印的 score 分布重新调整阈值。")