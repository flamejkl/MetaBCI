# offline_pseudo_online_eval.py
# -*- coding: utf-8 -*-
"""
离线伪在线评估脚本（滑窗 + AdvancedVoter）
1. 先固定窗口验证（确保对齐正确）
2. 再模拟真实在线流程：逐帧累积 → 滑窗预测 → 投票器决策
输出：在线准确率、平均决策时间、各方向准确率、ITR
"""
import os
import sys
import json
import numpy as np
import scipy.io as sio
import joblib
from collections import deque
from config import (
    SAMPLE_RATE,
    WINDOW_LEN_SAMPLES,
    VOTER_DECAY,
    VOTER_LOCK_FRAMES,
    VOTER_LOCK_DURATION,
    VOTER_THRESHOLD
)
from advanced_voter import AdvancedVoter

# ============================================================
#  配置参数
# ============================================================
MAT_FILE = r"G:\MetaBCI\MetaBCIWebGame\data\20260704_222224\full_experiment.mat"
MODEL_PATH = "self_ssvep_model.pkl"

# 枕区通道索引（与训练一致）
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]  # O1, O2, Oz, PO3, PO4, PO5, PO6, POz

# 模拟在线参数
STEP_SIZE_MS = 100                     # 每100ms到达一个数据包
STEP_SIZE_SAMPLES = int(SAMPLE_RATE * STEP_SIZE_MS / 1000)  # 25 samples
MAX_DECISION_TIME = 4.0                # 最多等待4秒，超时视为失败

# ============================================================
#  复用 extract_with_hw_trigger 的检测函数
# ============================================================
def detect_trigger_events(trigger_channel, threshold=0.5, min_high_duration=1):
    events = []
    i = 0
    n = len(trigger_channel)
    while i < n:
        if trigger_channel[i] < threshold:
            j = i + 1
            while j < n and trigger_channel[j] < threshold:
                j += 1
            if j < n:
                if j + min_high_duration <= n and np.all(trigger_channel[j:j + min_high_duration] >= threshold):
                    onset = j
                    value = int(round(trigger_channel[onset]))
                    if value in [1, 2, 3, 4]:
                        events.append((onset, value - 1))
                    while j < n and trigger_channel[j] >= threshold:
                        j += 1
                    i = j
                    continue
        i += 1
    return events

# ============================================================
#  加载模型和数据
# ============================================================
print("加载模型...")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ 模型加载成功: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

print(f"加载数据文件: {MAT_FILE}")
try:
    mat_data = sio.loadmat(MAT_FILE)
except Exception as e:
    print(f"❌ 加载 .mat 文件失败: {e}")
    sys.exit(1)

if 'eeg_data' not in mat_data:
    raise RuntimeError("文件中没有 'eeg_data' 字段")
eeg = mat_data['eeg_data']
print(f"原始 EEG 形状: {eeg.shape}")

# 从 Trigger 通道解析事件
trigger = eeg[-1, :]
event_indices = detect_trigger_events(trigger, threshold=0.5)
print(f"从 Trigger 通道解析到 {len(event_indices)} 个试次")
if not event_indices:
    raise RuntimeError("未检测到任何触发脉冲")

# ============================================================
#  预处理函数
# ============================================================
def preprocess_window(window):
    """与训练/在线保持一致：仅去均值"""
    return window - np.mean(window, axis=1, keepdims=True)

# ============================================================
#  固定窗口验证（快速确认对齐）
# ============================================================
print("\n===== 固定窗口验证（确保对齐正确）=====")
fixed_results = []
for start_idx, true_label in event_indices:
    if start_idx + WINDOW_LEN_SAMPLES > eeg.shape[1]:
        continue
    window_raw = eeg[:14, start_idx:start_idx + WINDOW_LEN_SAMPLES]
    window = window_raw[OCCIPITAL_INDICES, :]
    window = preprocess_window(window)
    X_input = window[np.newaxis, ...]

    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_input)[0]
    else:
        label = model.predict(X_input)[0]
        prob = np.zeros(4)
        if 0 <= label < 4:
            prob[label] = 1.0
        else:
            prob = np.ones(4) * 0.25
    decision = np.argmax(prob)
    fixed_results.append(decision == true_label)

fixed_acc = sum(fixed_results) / len(fixed_results) if fixed_results else 0
print(f"固定窗口准确率: {fixed_acc * 100:.2f}%")

if fixed_acc < 0.80:
    print("⚠️ 固定窗口准确率低于80%，请检查通道顺序或触发检测阈值后重试。")
    sys.exit(0)
else:
    print("✅ 对齐验证通过，开始滑窗模拟...\n")

# ============================================================
#  模拟单个试次（滑窗 + 投票器）
# ============================================================
def simulate_trial(eeg_data, start_idx, true_label, model, voter):
    """
    模拟真实在线数据流：
    - 从 start_idx 开始逐包累积数据
    - 每当缓冲区达到窗口长度，进行一次预测
    - 更新投票器，若锁定则立即返回
    """
    voter.reset()
    n_chans = eeg_data.shape[0] - 1  # 去掉Trigger通道
    buffer = deque(maxlen=WINDOW_LEN_SAMPLES)  # 自动丢弃最旧样本

    pos = start_idx
    max_pos = min(start_idx + int(MAX_DECISION_TIME * SAMPLE_RATE), eeg_data.shape[1])

    while pos < max_pos:
        # 取出一个数据包（STEP_SIZE_SAMPLES个点）
        end_pos = min(pos + STEP_SIZE_SAMPLES, max_pos)
        chunk = eeg_data[:n_chans, pos:end_pos]  # 取前14通道
        for t in range(chunk.shape[1]):
            buffer.append(chunk[:, t])  # 逐样本加入
        pos = end_pos

        # 如果缓冲区满了，进行预测
        if len(buffer) == WINDOW_LEN_SAMPLES:
            # 组装窗口 (channels, samples)
            window = np.array(buffer).T
            # 提取枕区通道
            window = window[OCCIPITAL_INDICES, :]
            # 预处理
            window = preprocess_window(window)
            X_input = window[np.newaxis, ...]

            # 预测
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_input)[0]
            else:
                label = model.predict(X_input)[0]
                prob = np.zeros(4)
                if 0 <= label < 4:
                    prob[label] = 1.0
                else:
                    prob = np.ones(4) * 0.25

            # 更新投票器（时间戳 = 已用时间）
            elapsed = (pos - start_idx) / SAMPLE_RATE
            decision, conf = voter.update(prob, timestamp=elapsed)

            # 如果投票器输出有效决策
            if decision is not None:
                return decision, elapsed, conf, prob.tolist()

    # 超时未决
    return None, None, 0.0, [0.0] * 4

# ============================================================
#  主循环
# ============================================================
print("===== 开始滑窗+投票器伪在线评估 =====")
print(f"总试次数: {len(event_indices)}")
print(f"步长: {STEP_SIZE_MS}ms, 窗口: {WINDOW_LEN_SAMPLES/SAMPLE_RATE:.1f}s")
print(f"投票器参数: decay={VOTER_DECAY}, lock_frames={VOTER_LOCK_FRAMES}, threshold={VOTER_THRESHOLD}")
print("-" * 60)

voter = AdvancedVoter(
    decay=VOTER_DECAY,
    lock_frames=VOTER_LOCK_FRAMES,
    lock_duration=VOTER_LOCK_DURATION,
    threshold=VOTER_THRESHOLD
)

results = []
for i, (start_idx, true_label) in enumerate(event_indices):
    if start_idx + WINDOW_LEN_SAMPLES > eeg.shape[1]:
        continue

    decoded, decision_time, conf, all_conf = simulate_trial(
        eeg, start_idx, true_label, model, voter
    )

    results.append({
        'trial': i,
        'true': int(true_label),
        'decoded': int(decoded) if decoded is not None else -1,
        'time': decision_time if decision_time is not None else MAX_DECISION_TIME,
        'confidence': conf,
        'all_conf': all_conf
    })

    if (i + 1) % 20 == 0:
        print(f"已处理 {i+1}/{len(event_indices)} 试次")

# ============================================================
#  统计指标
# ============================================================
print("\n" + "=" * 60)
print("📊 滑窗+投票器 伪在线评估结果")
print("=" * 60)

total = len(results)
successful = [r for r in results if r['decoded'] != -1]
timeout = [r for r in results if r['decoded'] == -1]

correct = [r for r in successful if r['decoded'] == r['true']]
acc = len(correct) / len(successful) if successful else 0
avg_time = np.mean([r['time'] for r in successful]) if successful else 0

print(f"总试次数: {total}")
print(f"成功决策: {len(successful)} ({len(successful)/total*100:.1f}%)")
print(f"超时/未决: {len(timeout)} ({len(timeout)/total*100:.1f}%)")
print(f"\n✅ 在线准确率 (成功决策中): {acc * 100:.2f}%")
print(f"⏱️ 平均决策时间: {avg_time * 1000:.1f} ms")

# 各方向准确率
print("\n--- 各方向准确率 ---")
for label in range(4):
    dir_name = ['上', '下', '左', '右'][label]
    trials_label = [r for r in successful if r['true'] == label]
    correct_label = [r for r in trials_label if r['decoded'] == label]
    if trials_label:
        print(f"{dir_name}: {len(correct_label)}/{len(trials_label)} ({len(correct_label)/len(trials_label)*100:.2f}%)")
    else:
        print(f"{dir_name}: 无试次")

# 决策时间分布
if successful:
    times = [r['time'] for r in successful]
    print(f"\n决策时间分布:")
    print(f"  最快: {min(times)*1000:.0f} ms")
    print(f"  最慢: {max(times)*1000:.0f} ms")
    print(f"  标准差: {np.std(times)*1000:.0f} ms")

# 计算 ITR (Information Transfer Rate)
if acc > 0 and avg_time > 0:
    N = 4
    if acc == 1.0:
        # 避免 log2(0) 错误
        itr = (np.log2(N) + acc * np.log2(acc)) / avg_time
    else:
        itr = (np.log2(N) + acc * np.log2(acc) + (1 - acc) * np.log2((1 - acc) / (N - 1))) / avg_time
    print(f"\n📈 信息传输率 (ITR): {itr:.2f} bits/s")

# 保存详细结果
with open("pseudo_online_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n详细结果已保存至: pseudo_online_results.json")

# 评价
if acc >= 0.85:
    print("\n🎉 在线模拟准确率优秀！该模型+投票器组合对实时数据流具备良好的鲁棒性。")
elif acc >= 0.70:
    print("\n👍 在线模拟准确率良好，可以尝试调整投票器参数或减小步长进一步优化。")
else:
    print("\n⚠️ 在线模拟准确率偏低，建议：")
    print("  1. 增大 VOTER_LOCK_FRAMES（如4→5）")
    print("  2. 适当提高 VOTER_THRESHOLD（如0.3→0.35）")
    print("  3. 或缩短 STEP_SIZE_MS（如100→50）")