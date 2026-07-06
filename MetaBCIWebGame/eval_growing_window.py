# eval_growing_window.py
# -*- coding: utf-8 -*-
"""
统计 Growing Window 各时间点的判别分数分布，并支持阈值搜索。
"""
import os
import re
import json
import numpy as np
import joblib
from collections import defaultdict
from sklearn.metrics import accuracy_score

# ============================================================
#  配置
# ============================================================
from config import OFFLINE_DATA_ROOT
DATA_ROOT = OFFLINE_DATA_ROOT
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]
SAMPLE_RATE = 250
WINDOW_LENGTHS = [125, 250, 375, 500]  # 模型对应的窗口长度

MODEL_PATHS = {
    125: "model_125.pkl",
    250: "model_250.pkl",
    375: "model_375.pkl",
    500: "self_ssvep_model.pkl"
}

# 检测步长（点）和对应时间
CHECK_STEP = 25  # 100ms
CHECK_TIMES = np.arange(0.5, 2.05, CHECK_STEP / SAMPLE_RATE)  # 0.5, 0.6, 0.7, ... 2.0s

# ============================================================
#  加载模型
# ============================================================
print("加载模型...")
models = {}
for L, path in MODEL_PATHS.items():
    try:
        models[L] = joblib.load(path)
        print(f"✅ 加载 {path} (长度 {L} 点)")
    except Exception as e:
        print(f"❌ 加载 {path} 失败: {e}")
        exit(1)


# ============================================================
#  加载数据
# ============================================================
def load_all_trials(root):
    X_list, y_list = [], []
    for label in range(4):
        dir_path = os.path.join(root, str(label + 1))
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy') and 'offset' not in f]
        for fname in files:
            data = np.load(os.path.join(dir_path, fname))
            data = data[OCCIPITAL_INDICES, :]
            X_list.append(data)
            y_list.append(label)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


print("\n加载数据...")
X, y_true = load_all_trials(DATA_ROOT)
print(f"加载 {X.shape[0]} 个试次")


# ============================================================
#  预处理
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
#  收集所有试次的时间序列记录
# ============================================================
print("\n===== 收集 Growing Window 时间序列 =====")
print(f"检测步长: {CHECK_STEP} 点 ({CHECK_STEP / SAMPLE_RATE * 1000:.0f}ms)")
print(f"检测时间点: {len(CHECK_TIMES)} 个 (0.5s ~ 2.0s)")
print("-" * 60)

all_trial_records = []  # 每个元素是一个试次的完整记录

for trial_idx in range(X.shape[0]):
    data = X[trial_idx]
    true_label = y_true[trial_idx]
    trial_record = []

    # 模拟 Growing Window：从 0.5s 开始，到 2.0s 结束
    for pos in range(int(0.5 * SAMPLE_RATE), data.shape[1] + 1, CHECK_STEP):
        L = pos
        current_time = pos / SAMPLE_RATE

        # 选择对应长度的模型
        model_len = None
        for wl in sorted(WINDOW_LENGTHS):
            if wl <= L:
                model_len = wl
            else:
                break
        if model_len is None:
            continue

        # 取前 L 个点（从试次起始累积）
        window = data[:, :L]
        # 取模型需要的长度
        window_trim = window[:, -model_len:] if L >= model_len else window
        if window_trim.shape[1] < model_len:
            continue

        window_trim = preprocess(window_trim)
        scores = get_scores(window_trim, models[model_len])

        max_score = np.max(scores)
        sorted_scores = np.sort(scores)[::-1]
        margin = sorted_scores[0] - sorted_scores[1]
        decision = np.argmax(scores)
        correct = (decision == true_label)

        trial_record.append({
            'time': current_time,
            'model_len': model_len,
            'scores': scores.copy(),
            'max': max_score,
            'margin': margin,
            'decision': decision,
            'correct': correct,
            'true_label': true_label
        })

    all_trial_records.append(trial_record)

    if (trial_idx + 1) % 50 == 0:
        print(f"已处理 {trial_idx + 1}/{X.shape[0]} 试次")

print(f"共收集 {len(all_trial_records)} 个试次，每个试次约 {len(all_trial_records[0])} 个检测点")

# ============================================================
#  模式选择
# ============================================================
print("\n" + "=" * 60)
print("请选择模式:")
print("  1. 统计模式: 打印各时间点的分数分布")
print("  2. 搜索模式: 自动搜索最优阈值")
mode = input("请输入模式编号 (1/2): ").strip()

# ============================================================
#  模式1：统计分布
# ============================================================
if mode == '1':
    print("\n" + "=" * 60)
    print("📊 各时间点判别分数统计")
    print("=" * 60)

    # 按时间点分组
    time_groups = defaultdict(list)
    for record in all_trial_records:
        for rec in record:
            time_groups[rec['time']].append(rec)

    for time_point in sorted(time_groups.keys()):
        records = time_groups[time_point]
        correct_records = [r for r in records if r['correct']]
        wrong_records = [r for r in records if not r['correct']]

        print(f"\n--- {time_point:.2f}s ---")
        if correct_records:
            max_vals = [r['max'] for r in correct_records]
            margin_vals = [r['margin'] for r in correct_records]
            print(f"  正确样本 ({len(correct_records)}): max={np.mean(max_vals):.3f}±{np.std(max_vals):.3f}, "
                  f"margin={np.mean(margin_vals):.3f}±{np.std(margin_vals):.3f}")
        if wrong_records:
            max_vals = [r['max'] for r in wrong_records]
            margin_vals = [r['margin'] for r in wrong_records]
            print(f"  错误样本 ({len(wrong_records)}): max={np.mean(max_vals):.3f}±{np.std(max_vals):.3f}, "
                  f"margin={np.mean(margin_vals):.3f}±{np.std(margin_vals):.3f}")

    print("\n✅ 统计完成")

# ============================================================
#  模式2：阈值搜索
# ============================================================
elif mode == '2':
    print("\n" + "=" * 60)
    print("🔍 开始阈值搜索")
    print("=" * 60)

    # ---------- 搜索参数 ----------
    # 不同时间点的 margin 阈值（分别搜索）
    MARGIN_RANGE = np.arange(0.05, 1.0, 0.05)
    # max 阈值
    MAX_RANGE = np.arange(0.1, 1.0, 0.05)
    # 连续一致次数
    CONSECUTIVE_RANGE = [1, 2, 3]

    print(f"margin 搜索范围: {MARGIN_RANGE[0]:.2f} ~ {MARGIN_RANGE[-1]:.2f}, 步长 0.05")
    print(f"max 搜索范围: {MAX_RANGE[0]:.2f} ~ {MAX_RANGE[-1]:.2f}, 步长 0.05")
    print(f"连续次数: {CONSECUTIVE_RANGE}")
    print("-" * 60)


    def simulate_decision(trial_record, margin_threshold, max_threshold, consecutive_required):
        """
        模拟单个试次的在线决策
        """
        last_decision = None
        consecutive_count = 0

        for rec in trial_record:
            decision = rec['decision']
            margin = rec['margin']
            max_score = rec['max']
            current_time = rec['time']

            # 检查条件
            if margin < margin_threshold or max_score < max_threshold:
                consecutive_count = 0
                last_decision = None
                continue

            # 连续一致检查
            if decision == last_decision:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_decision = decision

            if consecutive_count >= consecutive_required:
                return current_time, decision, rec['max']

        # 未提前停止，使用最后一个时间点
        last_rec = trial_record[-1]
        return last_rec['time'], last_rec['decision'], last_rec['max']


    def evaluate_params(margin_threshold, max_threshold, consecutive_required):
        """评估一组参数"""
        decisions = []
        stop_times = []

        for trial_record in all_trial_records:
            true_label = trial_record[0]['true_label']
            stop_time, decision, _ = simulate_decision(
                trial_record, margin_threshold, max_threshold, consecutive_required
            )
            decisions.append(decision)
            stop_times.append(stop_time)

        true_labels = [r[0]['true_label'] for r in all_trial_records]
        accuracy = accuracy_score(true_labels, decisions)
        avg_time = np.mean(stop_times)
        std_time = np.std(stop_times)
        early_stop_rate = np.mean([t < 2.0 for t in stop_times])

        # ITR
        N = 4
        if accuracy > 0 and avg_time > 0:
            if accuracy == 1.0:
                itr = np.log2(N) / avg_time
            else:
                itr = (np.log2(N) + accuracy * np.log2(accuracy) +
                       (1 - accuracy) * np.log2((1 - accuracy) / (N - 1))) / avg_time
        else:
            itr = 0

        return {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'std_time': std_time,
            'early_stop_rate': early_stop_rate,
            'itr': itr,
            'decisions': decisions,
            'stop_times': stop_times
        }


    # ---------- 执行搜索 ----------
    results = []
    total_combinations = len(MARGIN_RANGE) * len(MAX_RANGE) * len(CONSECUTIVE_RANGE)
    count = 0

    for margin_th in MARGIN_RANGE:
        for max_th in MAX_RANGE:
            for cons in CONSECUTIVE_RANGE:
                count += 1
                result = evaluate_params(margin_th, max_th, cons)
                results.append({
                    'margin': margin_th,
                    'max': max_th,
                    'consecutive': cons,
                    'accuracy': result['accuracy'],
                    'avg_time': result['avg_time'],
                    'itr': result['itr'],
                    'early_stop_rate': result['early_stop_rate']
                })

                if count % 50 == 0:
                    print(f"进度: {count}/{total_combinations}")

    # ---------- 输出结果 ----------
    print("\n" + "=" * 60)
    print("📊 搜索结果")
    print("=" * 60)

    # 筛选准确率 >= 98%
    high_acc = [r for r in results if r['accuracy'] >= 0.98]

    if high_acc:
        # 按 ITR 降序排列
        sorted_results = sorted(high_acc, key=lambda x: x['itr'], reverse=True)
        top_results = sorted_results[:10]

        print(f"\n前 10 个最优结果 (准确率 ≥ 98%, 按 ITR 排序):")
        print("-" * 60)
        for i, r in enumerate(top_results):
            print(f"{i + 1}. margin={r['margin']:.2f}, max={r['max']:.2f}, "
                  f"连续={r['consecutive']}次")
            print(f"   准确率: {r['accuracy'] * 100:.2f}%")
            print(f"   平均决策时间: {r['avg_time'] * 1000:.0f} ms")
            print(f"   ITR: {r['itr']:.3f} bits/s")
            print(f"   提前停止率: {r['early_stop_rate'] * 100:.1f}%")
            print()
    else:
        # 如果没有达到 98%，输出准确率最高的结果
        best = max(results, key=lambda x: x['accuracy'])
        print(f"未找到准确率 ≥ 98% 的组合。")
        print(f"最佳组合: margin={best['margin']:.2f}, max={best['max']:.2f}, 连续={best['consecutive']}次")
        print(f"准确率: {best['accuracy'] * 100:.2f}%")
        print(f"平均决策时间: {best['avg_time'] * 1000:.0f} ms")
        print(f"ITR: {best['itr']:.3f} bits/s")

    # 保存所有结果
    with open('threshold_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n所有结果已保存至 threshold_search_results.json")

else:
    print("无效模式，退出。")