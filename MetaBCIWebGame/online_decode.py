# online_decode.py
# -*- coding: utf-8 -*-
import time
import numpy as np
import joblib
import os
import scipy.io as sio
from scipy.signal import butter, filtfilt
import config
from config import SAMPLE_RATE, CHANNELS, DYNAMIC_THRESHOLD, L_FREQ, H_FREQ

# ========== 随机模型（用于测试） ==========
class DummyModel:
    def predict(self, X):
        return np.random.randint(0, 4, size=len(X))
    def transform(self, X):
        return np.random.randn(len(X), 4)

# ========== 动态停止解码器（Growing Window 版本） ==========
class DynamicStoppingDecoder:
    def __init__(self, model_dict=None):
        """
        model_dict: dict, 键为窗口长度（采样点数），值为对应的模型实例。
                    例如 {125: model_125, 250: model_250, 375: model_375, 500: model_500}
        """
        if model_dict is None:
            # 默认加载预设的四个模型
            model_dict = {}
            for L in [125, 250, 375, 500]:
                fname = f"model_{L}.pkl" if L != 500 else "self_ssvep_model.pkl"
                try:
                    model_dict[L] = joblib.load(fname)
                    print(f"✅ 加载模型 {fname} (长度 {L} 点)")
                except:
                    print(f"⚠️ 无法加载 {fname}，使用 DummyModel")
                    model_dict[L] = DummyModel()
        self.model_dict = model_dict
        self.window_lens = sorted(model_dict.keys())  # [125, 250, 375, 500]
        self.sample_rate = SAMPLE_RATE
        self.buffer = []
        self.ema = None
        self.consecutive = 0
        self.last_stable_decision = None
        # 停止条件参数
        self.max_threshold = 0.9          # 最大置信度阈值
        self.margin_threshold = 0.3       # 首末差值阈值
        self.ema_decay = 0.7              # EMA 衰减系数
        self.require_consecutive = 2      # 连续稳定次数要求

        print(f"动态停止解码器初始化，支持窗口长度: {self.window_lens}")

    def _preprocess(self, window):
        """与训练时一致的预处理：去均值 + 窄带增强（可选）"""
        window = window - np.mean(window, axis=1, keepdims=True)
        # 窄带增强（针对 16.5 Hz，可保留或注释）
        from scipy.signal import butter, sosfilt
        fs = self.sample_rate
        sos = butter(4, [15.5, 17.5], btype='bandpass', fs=fs, output='sos')
        filtered = sosfilt(sos, window, axis=-1)
        window = window + 0.5 * filtered
        return window

    def _get_prob(self, window, model):
        """
        使用模型的 transform 方法输出原始分数，并转为概率分布（softmax）
        """
        # 预处理
        window = self._preprocess(window)
        X_input = window[np.newaxis, ...]  # (1, channels, samples)
        try:
            # 优先使用 transform 获取原始分数
            if hasattr(model, 'transform'):
                raw_scores = model.transform(X_input)[0]  # (n_classes,)
            else:
                # 若模型无 transform（如 DummyModel），回退到 predict
                pred_label = model.predict(X_input)[0]
                raw_scores = np.zeros(4)
                if 0 <= pred_label < 4:
                    raw_scores[pred_label] = 1.0
                else:
                    raw_scores = np.ones(4) * 0.25
        except Exception as e:
            print(f"预测异常: {e}")
            raw_scores = np.ones(4) * 0.25

        # softmax 归一化
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        prob = exp_scores / np.sum(exp_scores)
        return prob

    def feed(self, sample):
        """
        输入一个样本（单通道单时间点，长度为 n_channels 的向量），更新内部缓冲区，
        若达到某个窗口长度，则进行预测并判断是否满足提前停止条件。
        返回 (decision, confidence, current_length) 或 (None, 0.0, current_length)
        """
        self.buffer.append(sample)
        current_len = len(self.buffer)

        # 检查是否达到了某个模型长度
        for L in self.window_lens:
            if current_len == L:  # 精确匹配（也可用 >=，但为保证起点一致，我们用 ==）
                # 取最后 L 个点组成窗口
                window = np.array(self.buffer[-L:]).T  # (channels, L)
                prob = self._get_prob(window, self.model_dict[L])

                max_prob = np.max(prob)
                # 计算 margin（首末差值）
                sorted_probs = np.sort(prob)[::-1]
                margin = sorted_probs[0] - sorted_probs[1]

                # 更新 EMA
                if self.ema is None:
                    self.ema = prob
                else:
                    self.ema = self.ema_decay * self.ema + (1 - self.ema_decay) * prob

                # 判断是否稳定（EMA 与当前预测一致）
                current_decision = np.argmax(prob)
                ema_decision = np.argmax(self.ema)

                if current_decision == ema_decision:
                    self.consecutive += 1
                else:
                    self.consecutive = 0

                # 综合判断是否提前停止
                if max_prob > self.max_threshold and margin > self.margin_threshold and self.consecutive >= self.require_consecutive:
                    # 稳定输出
                    self.last_stable_decision = current_decision
                    return current_decision, max_prob, current_len

        # 若未达到任何模型长度，或条件不满足，返回 None
        return None, 0.0, current_len

    def reset(self):
        """重置解码器状态，用于下一个试次"""
        self.buffer = []
        self.ema = None
        self.consecutive = 0
        self.last_stable_decision = None

    # ---------- 兼容旧接口（若只使用一个模型） ----------
    def predict_window(self, window):
        """
        单窗口预测（保留原接口，用于固定窗口模式）
        """
        if window.shape[1] < self.window_lens[-1]:
            return None, 0.0, [0.0]*4
        if window.shape[1] > self.window_lens[-1]:
            window = window[:, :self.window_lens[-1]]
        # 使用最长模型
        model = self.model_dict[self.window_lens[-1]]
        prob = self._get_prob(window, model)
        max_conf = np.max(prob)
        if max_conf >= DYNAMIC_THRESHOLD:
            decision = np.argmax(prob)
            return decision, max_conf, prob.tolist()
        else:
            return None, max_conf, prob.tolist()

# ========== 模拟数据流（用于测试） ==========
def get_eeg_stream():
    while True:
        sample = np.random.randn(14) * 0.1
        yield sample
        time.sleep(1.0 / SAMPLE_RATE)

# ========== 主循环（示例） ==========
def online_decode_loop(callback, stop_flag=None, model_dict=None, offline_replay=False):
    """
    在线解码主循环，使用 GrowingWindowDecoder
    """
    if model_dict is None:
        # 尝试加载预设模型
        model_dict = {}
        for L in [125, 250, 375, 500]:
            fname = f"model_{L}.pkl" if L != 500 else "self_ssvep_model.pkl"
            try:
                model_dict[L] = joblib.load(fname)
                print(f"加载模型 {fname}")
            except:
                model_dict[L] = DummyModel()
                print(f"无法加载 {fname}，使用随机模型")

    decoder = DynamicStoppingDecoder(model_dict)

    if offline_replay:
        print("离线回放模式暂未实现，使用模拟数据流")
    else:
        print("开始实时解码...")
        for chunk in get_eeg_stream():
            if stop_flag and stop_flag():
                break
            decision, conf, length = decoder.feed(chunk)
            if decision is not None:
                callback(decision, conf, length)
                # 复位解码器（如需每个指令独立，可保留；若连续流则可不重置）
                # decoder.reset()