# extract_with_hw_trigger.py
import os
import re
import numpy as np
import scipy.io as sio

# ===== 配置 =====
MAT_FILE = r"D:\pycharm\PyCharm 2026.1\my-projects\MetaBCI\MetaBCIWebGame\data\20260703_222223\full_experiment.mat"  # 修改为你的 .mat 文件路径
OUTPUT_ROOT = r"D:\pycharm\PyCharm 2026.1\my-projects\MetaBCI\data_self_multi_offset"  # 建议新建一个目录，避免覆盖原数据
WINDOW_SAMPLES = 500
TRIGGER_THRESHOLD = 0.5

# ========== 新增：多偏移参数 ==========
# 生成偏移样本的最大偏移量（采样点），建议覆盖在线滑窗步长（100ms=25点）的 2~3 倍
MAX_OFFSET_SAMPLES = 125  # 125 点 = 500ms，覆盖 0, 100, 200, 300, 400, 500ms
OFFSET_STEP = 25          # 步长 = 25 点 = 100ms（与在线模拟步长一致）

# 生成偏移列表：0, 25, 50, 75, 100, 125
OFFSET_LIST = list(range(0, MAX_OFFSET_SAMPLES + 1, OFFSET_STEP))

# ===== 增强的检测函数（保持不变） =====
def detect_trigger_events(trigger_channel, threshold=0.5, min_high_duration=1):
    """
    检测 Trigger 通道中的脉冲值（1~4 为开始，5 为结束）
    返回列表，每个元素为 (起始样本索引, 方向标签)
    """
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
                        events.append((onset, value - 1))  # 方向标签 0~3
                    while j < n and trigger_channel[j] >= threshold:
                        j += 1
                    i = j
                    continue
        i += 1
    return events

def get_existing_max_index(folder):
    """
    获取指定文件夹中已有 hw_trial_*.npy 文件的最大序号（忽略 _offset 后缀），若无则返回 -1
    """
    if not os.path.isdir(folder):
        return -1
    max_idx = -1
    # 修改正则：匹配 hw_trial_数字，后面可能有 _offset数字，但只提取基础数字
    pattern = re.compile(r'hw_trial_(\d+)(?:_offset\d+)?\.npy')
    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx

def main():
    print("开始执行多偏移窗口分割脚本...")
    print(f"偏移列表: {OFFSET_LIST} (共 {len(OFFSET_LIST)} 个偏移)")

    # 1. 检查文件是否存在
    if not os.path.exists(MAT_FILE):
        print(f"错误：文件不存在 - {MAT_FILE}")
        return

    print(f"正在加载文件: {MAT_FILE}")
    try:
        data = sio.loadmat(MAT_FILE)
    except Exception as e:
        print(f"加载 .mat 文件失败: {e}")
        return

    if 'eeg_data' not in data:
        print("错误：文件中没有 'eeg_data' 字段")
        return
    eeg = data['eeg_data']
    print(f"EEG 数据形状: {eeg.shape}")
    if eeg.shape[0] < 1:
        print("错误：EEG 数据为空")
        return

    trigger = eeg[-1, :]
    print(f"Trigger 通道长度: {len(trigger)}")
    print(f"Trigger 最大值: {trigger.max():.3f}, 最小值: {trigger.min():.3f}, 均值: {trigger.mean():.3f}")

    trigger_events = detect_trigger_events(trigger, threshold=TRIGGER_THRESHOLD)
    print(f"检测到 {len(trigger_events)} 个开始触发脉冲")

    if len(trigger_events) == 0:
        print("警告：未检测到任何开始脉冲，请检查阈值或 Trigger 信号形状。")
        print("尝试打印前 1000 个 Trigger 样本值:")
        print(trigger[:1000])
        return

    # 按方向分组统计
    print("\n各方向触发数量:")
    for label in range(4):
        count = sum(1 for e in trigger_events if e[1] == label)
        print(f"  方向 {label}: {count} 个")

    # 创建输出目录并获取已有最大索引
    start_counts = [0, 0, 0, 0]
    for label in range(4):
        folder = os.path.join(OUTPUT_ROOT, str(label + 1))
        os.makedirs(folder, exist_ok=True)
        max_existing = get_existing_max_index(folder)
        start_counts[label] = max_existing + 1

    print(f"\n已有文件最大索引：{start_counts}，新数据将从这些索引开始保存。")

    n_channels = 14
    counts = start_counts[:]
    total_samples = eeg.shape[1]

    # 记录每个方向实际生成的偏移窗口总数（用于后续核对）
    offset_counts = {label: 0 for label in range(4)}

    for onset, label in trigger_events:
        # ========== 核心改动：对每个偏移量生成一个窗口 ==========
        for offset in OFFSET_LIST:
            start = onset + offset
            end = start + WINDOW_SAMPLES
            if end > total_samples:
                # 如果超出总样本数，跳过该偏移（通常只有最大的偏移才会越界）
                print(f"警告：偏移 {offset} 点越界 (end={end} > {total_samples})，跳过")
                continue

            # 提取窗口
            epoch = eeg[:n_channels, start:end]

            # 保存为 .npy 文件
            out_dir = os.path.join(OUTPUT_ROOT, str(label + 1))
            # 文件名包含偏移量，便于追踪
            filename = f"hw_trial_{counts[label]:04d}_offset{offset:03d}.npy"
            np.save(os.path.join(out_dir, filename), epoch)

            offset_counts[label] += 1

        # 每处理完一个试次（所有偏移），递增该方向的计数
        counts[label] += 1

    print("\n===== 多偏移窗口提取完成 =====")
    for label in range(4):
        total_generated = offset_counts[label]
        print(f"方向 {label} (文件夹 {label + 1}): 共生成 {total_generated} 个偏移窗口")
        print(f"  每个试次生成 {len(OFFSET_LIST)} 个偏移，来自 {trigger_events[label] if label < len(trigger_events) else 0} 个试次")

    print(f"\n所有文件已保存至: {OUTPUT_ROOT}")
    print(f"总生成窗口数: {sum(offset_counts.values())}")

if __name__ == "__main__":
    main()