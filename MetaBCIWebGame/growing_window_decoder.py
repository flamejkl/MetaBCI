# growing_window_decoder.py
import numpy as np
import joblib
from collections import deque
from config import *

class GrowingWindowDecoder:
    def __init__(self, model_paths=GW_MODEL_PATHS):
        self.models = {}
        for L, path in model_paths.items():
            self.models[L] = joblib.load(path)
            print(f"✅ 加载模型 {path} (长度 {L} 点)")

        self.step = GW_CHECK_STEP
        self.margin_th = GW_MARGIN_THRESHOLD
        self.max_th = GW_MAX_THRESHOLD
        self.cons_req = GW_CONSECUTIVE_REQUIRED
        self.min_len = GW_MIN_LENGTH
        self.max_len = GW_MAX_LENGTH

        self.buffer = []          # 存储从刺激开始累积的样本（列表，每个元素是 (channels,)）
        self.history = deque(maxlen=self.cons_req)
        self.last_decision = None
        self.consecutive_count = 0
        self.sample_rate = SAMPLE_RATE
        self.occipital_indices = OCCIPITAL_INDICES  # 你的枕区通道索引

        # 为了性能，预先提取模型长度列表
        self.model_lengths = sorted(self.models.keys())

    def feed(self, sample):
        """
        喂入一个样本（长度为 channels 的 np.array）
        返回 (decision, confidence, current_time) 或 (None, 0.0, current_time)
        """
        self.buffer.append(sample)
        L = len(self.buffer)

        # 如果当前累积长度小于最小检测长度，直接返回
        if L < self.min_len:
            return None, 0.0, L / self.sample_rate

        # 只在检测步长检查（模拟每 100ms 判断一次）
        if L % self.step != 0:
            return None, 0.0, L / self.sample_rate

        # 选择 <= L 的最大模型
        model_len = None
        for wl in self.model_lengths:
            if wl <= L:
                model_len = wl
            else:
                break
        if model_len is None:
            return None, 0.0, L / self.sample_rate

        # 构建窗口：取前 L 个点（Growing Window）
        # 注意：self.buffer 是 list of (channels,)，需要转置为 (channels, L)
        window_full = np.array(self.buffer[:L]).T   # (channels, L)
        # 取该模型需要的长度（通常是最近 model_len 个点，但这里因为是 Growing，我们取整个窗口的末尾 model_len 个点）
        # 然而对于 Growing Window，我们更希望使用前 model_len 个点（即刺激开始后的 model_len 点）
        # 为了与训练一致，我们使用前 model_len 个点
        window_trim = window_full[:, :model_len]   # 从开头取

        # 提取枕区通道（如果 window 已经是枕区，可以省略）
        if window_trim.shape[0] != len(self.occipital_indices):
            window_trim = window_trim[self.occipital_indices, :]

        # 预处理（去均值 + 窄带增强）
        window_trim = self._preprocess(window_trim)

        # 预测
        X_input = window_trim[np.newaxis, ...]
        scores = self.models[model_len].transform(X_input)[0]  # (4,)

        # 计算 margin 和 max
        top2 = np.partition(scores, -2)[-2:]
        margin = top2.max() - top2.min()
        max_score = np.max(scores)
        decision = np.argmax(scores)

        # 判断停止条件
        if margin > self.margin_th and max_score > self.max_th:
            self.history.append(decision)
            if len(self.history) == self.cons_req and len(set(self.history)) == 1:
                # 提前停止
                return decision, max_score, L / self.sample_rate
        else:
            self.history.clear()

        return None, 0.0, L / self.sample_rate

    def reset(self):
        """重置解码器状态（每个试次开始前调用）"""
        self.buffer = []
        self.history.clear()
        self.last_decision = None
        self.consecutive_count = 0

    def _preprocess(self, window):
        """与训练一致的预处理"""
        window = window - np.mean(window, axis=1, keepdims=True)
        from scipy.signal import butter, sosfilt
        fs = self.sample_rate
        sos = butter(4, [15.5, 17.5], btype='bandpass', fs=fs, output='sos')
        filtered = sosfilt(sos, window, axis=-1)
        window = window + 0.5 * filtered
        return window