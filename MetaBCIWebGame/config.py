# config.py
# -*- coding: utf-8 -*-

# ========== 刺激参数 ==========
STIM_FREQS = [8.25, 11.0, 13.75, 16.5]      # Hz，对应 [上, 下, 左, 右]
STIM_WINDOW_SIZE = (800, 600)

# ========== EEG采集参数 ==========
SAMPLE_RATE = 250
ONLINE_SAMPLE_RATE = 250
DECIMATION_FACTOR = 1
CHANNELS = ['Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO5', 'PO6', 'POz', 'P3', 'P4', 'P7', 'P8']
L_FREQ = 8
H_FREQ = 30

# ========== 模型参数 ==========
MODEL_PATH = "self_ssvep_model.pkl"

# ========== 动态停止参数 ==========
DYNAMIC_STOPPING_WINDOWS = [0.4, 0.6, 0.8, 1.0]
DYNAMIC_THRESHOLD = 0.75

# ========== 高级投票器参数 ==========
VOTER_DECAY = 0.8
VOTER_LOCK_FRAMES = 3
VOTER_LOCK_DURATION = 0.5
VOTER_THRESHOLD = 0.5

# ========== WebSocket参数 ==========
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# ========== Neuracle 设备参数 ==========
NEURACLE_IP = "127.0.0.1"
NEURACLE_PORT = 8712
SERIAL_PORT = "COM3"

# ========== 窗口参数 ==========
WINDOW_LEN_SEC = 2.0
WINDOW_LEN_SAMPLES = int(SAMPLE_RATE * WINDOW_LEN_SEC)          # 500
RAW_WINDOW_SAMPLES = int(ONLINE_SAMPLE_RATE * WINDOW_LEN_SEC)   # 2000

# ========== 离线模拟数据路径 ==========
OFFLINE_DATA_ROOT = r"D:\pyproject\MetaBCI\data_self"   # 存放 .npy 文件的目录

# ========== 离线模拟超时测试参数 ==========
OFFLINE_TIMEOUT_PROB = 0.2      # 每个试次 20% 概率触发超时
OFFLINE_TIMEOUT_DELAY = 1.0

# config.py 末尾添加
FIXED_WINDOW_MODE = True   # True: 固定窗口模式（演示推荐）; False: 滑动窗口模式（用于连续控制）

# ========== Growing Window 动态停止参数 ==========
GW_MODEL_PATHS = {
    125: "model_125.pkl",
    250: "model_250.pkl",
    375: "model_375.pkl",
    500: "self_ssvep_model.pkl"
}
GW_CHECK_STEP = 25          # 检测步长（采样点），25 = 100ms
GW_MIN_LENGTH = 125         # 0.5s 开始检测
GW_MAX_LENGTH = 500         # 2.0s 强制停止
GW_MARGIN_THRESHOLD = 0.35
GW_MAX_THRESHOLD = 0.75
GW_CONSECUTIVE_REQUIRED = 1