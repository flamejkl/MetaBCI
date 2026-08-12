# -*- coding: utf-8 -*-
"""
FBCCA 在线解码器 — 无训练 SSVEP 解码，替代 FBTDCA GrowingWindowDecoder。

接口与 GrowingWindowDecoder 兼容: feed(sample) → (decision, confidence, t)
"""
import numpy as np
from collections import deque
from scipy.signal import cheby1, sosfiltfilt


class FBCCADecoder:
    """FBCCA 在线解码器，支持递增窗口动态停止。"""

    def __init__(
        self,
        freqs=(8.25, 11.0, 13.75, 16.5),
        sample_rate=250,
        n_channels=8,
        n_harmonics=2,
        n_subbands=3,
        weight_a=1.0,
        weight_b=0.0,
        filter_order=4,
        step=25,
        min_len=250,           # 1.0s 起检 (FBTDCA 是 125)
        max_len=500,           # 2.0s 强制输出
        margin_th=0.10,
        cons_req=1,
        reg=1e-4,
    ):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.n_freqs = len(freqs)
        self.freqs = np.array(freqs)
        self.step = step
        self.min_len = min_len
        self.max_len = max_len
        self.margin_th = margin_th
        self.cons_req = cons_req
        self.reg = reg
        self.n_harmonics = n_harmonics

        # ---- 滤波器组 ----
        self.sos_list = [
            cheby1(filter_order, 0.5, [8 * (i + 1), 90],
                   btype='bandpass', output='sos', fs=sample_rate)
            for i in range(n_subbands)
        ]
        self.filterweights = np.array(
            [(i + 1) ** (-weight_a) + weight_b for i in range(n_subbands)]
        )

        # ---- 环缓冲区 ----
        self._buf = np.zeros((n_channels, max_len), dtype=np.float64)
        self._write = 0
        self._total = 0

        # ---- 历史 ----
        self.history = deque(maxlen=cons_req)

        # ---- 缓存 ----
        self._last_scores = None

    # ------------------------------------------------------------------
    def feed(self, sample):
        """喂入一个样本 (n_channels,)，返回 (decision, confidence, t)。"""
        self._buf[:, self._write] = sample
        self._write = (self._write + 1) % self.max_len
        if self._total < self.max_len:
            self._total += 1

        L = self._total
        if L < self.min_len:
            return None, 0.0, L / self.sample_rate
        if L % self.step != 0:
            return None, 0.0, L / self.sample_rate

        window = self._extract(L)
        scores = self._compute_scores(window)
        self._last_scores = scores

        top2 = np.partition(scores, -2)[-2:]
        margin = top2.max() - top2.min()
        decision = np.argmax(scores)

        if margin > self.margin_th:
            self.history.append(decision)
            if len(self.history) == self.cons_req and len(set(self.history)) == 1:
                return decision, np.max(scores), L / self.sample_rate
        else:
            self.history.clear()

        if L >= self.max_len:
            return np.argmax(scores), np.max(scores), self.max_len / self.sample_rate

        return None, 0.0, L / self.sample_rate

    # ------------------------------------------------------------------
    def reset(self):
        """试次重置（保留滤波器状态）。"""
        self._buf.fill(0.0)
        self._write = 0
        self._total = 0
        self.history.clear()

    def reset_normaliser(self):
        """FBCCA 无 EMA 状态，接口兼容占位。"""
        pass

    def slide(self, n=None):
        if n is None:
            n = self.step
        if n >= self._total:
            self.reset()
            return
        self._write = (self._write - n) % self.max_len
        self._total -= n

    # ------------------------------------------------------------------
    def _extract(self, length):
        """从环缓冲区取连续 length 个样本。"""
        if self._write >= length:
            return self._buf[:, self._write - length:self._write].copy()
        first = self.max_len - (length - self._write)
        return np.hstack([
            self._buf[:, first:],
            self._buf[:, :self._write],
        ])

    def _compute_scores(self, window):
        """FBCCA 打分: 子带滤波 + CCA 相关系数加权求和。"""
        n_samples = window.shape[1]
        t = np.arange(n_samples) / self.sample_rate
        scores = np.zeros(self.n_freqs)

        for fi, f in enumerate(self.freqs):
            # 生成参考信号
            refs = []
            for h in range(1, self.n_harmonics + 1):
                refs.append(np.sin(2 * np.pi * f * h * t))
                refs.append(np.cos(2 * np.pi * f * h * t))
            Yf = np.array(refs)  # (2*Nh, n_samples)

            rho_sum = 0.0
            for sb, sos in enumerate(self.sos_list):
                Xf = sosfiltfilt(sos, window, axis=-1)
                rho = self._cca_corr(Xf, Yf)
                rho_sum += self.filterweights[sb] * rho
            scores[fi] = rho_sum

        return scores

    def _cca_corr(self, X, Y):
        """SVD 法计算 X 与 Y 的最大典型相关系数 (带正则化)。"""
        n_ch = X.shape[0]
        n_t = X.shape[1]
        X_c = X - X.mean(axis=1, keepdims=True)
        Y_c = Y - Y.mean(axis=1, keepdims=True)

        Cxx = X_c @ X_c.T / (n_t - 1) + self.reg * np.eye(n_ch)
        Cyy = Y_c @ Y_c.T / (n_t - 1) + self.reg * np.eye(Y.shape[0])
        Cxy = X_c @ Y_c.T / (n_t - 1)

        Ux, Sx, _ = np.linalg.svd(Cxx)
        Uy, Sy, _ = np.linalg.svd(Cyy)
        Cxx_is = Ux @ np.diag(1.0 / np.sqrt(np.maximum(Sx, 1e-10))) @ Ux.T
        Cyy_is = Uy @ np.diag(1.0 / np.sqrt(np.maximum(Sy, 1e-10))) @ Uy.T

        _, S, _ = np.linalg.svd(Cxx_is @ Cxy @ Cyy_is)
        return S[0] if len(S) > 0 else 0.0
