# growing_window_decoder.py
import numpy as np
import joblib
from collections import deque
from scipy.signal import butter, sosfilt
from config import (
    GW_MODEL_PATHS, GW_CHECK_STEP, GW_MIN_LENGTH, GW_MAX_LENGTH,
    GW_MARGIN_THRESHOLD, GW_MAX_THRESHOLD, GW_CONSECUTIVE_REQUIRED,
    SAMPLE_RATE, OCCIPITAL_INDICES
)


class GrowingWindowDecoder:
    """Growing-window SSVEP decoder with dynamic stopping.

    Uses a pre-allocated ring buffer and pre-computed filter coefficients
    to avoid repeated allocations and filter design in the hot path.
    Supports slide() for continuous REALTIME decoding without unbounded growth.
    """

    def __init__(self, model_paths=None):
        if model_paths is None:
            model_paths = GW_MODEL_PATHS

        # ---- load models ----
        self.models = {}
        for L, path in model_paths.items():
            self.models[L] = joblib.load(path)
            print(f"[GWDecoder] 加载模型 {path} (长度 {L} 点)")

        # ---- config ----
        self.step = GW_CHECK_STEP
        self.margin_th = GW_MARGIN_THRESHOLD
        self.max_th = GW_MAX_THRESHOLD
        self.cons_req = GW_CONSECUTIVE_REQUIRED
        self.min_len = GW_MIN_LENGTH
        self.max_len = GW_MAX_LENGTH
        self.sample_rate = SAMPLE_RATE
        self.n_chans = len(OCCIPITAL_INDICES)
        self.model_lengths = sorted(self.models.keys())

        # ---- pre-allocated ring buffer (channels, max_len) ----
        self._buf = np.zeros((self.n_chans, self.max_len), dtype=np.float64)
        self._start = 0     # oldest valid sample index (ring buffer offset)
        self._write = 0     # next write position (ring buffer offset)
        self._total = 0     # number of valid samples currently in buffer

        # ---- consecutive-decision history ----
        self.history = deque(maxlen=self.cons_req)

        # ---- pre-compute filter coefficients ONCE ----
        fs = self.sample_rate
        # Broad bandpass covering all four SSVEP frequencies (8–17 Hz with margins)
        self._sos = butter(4, [7, 18], btype='bandpass', fs=fs, output='sos')

        # ---- cache for forced-output at max_len to avoid redundant compute ----
        self._cached_window = None    # (model_len, preprocessed_window)
        self._cached_model_len = None

        print(f"[GWDecoder] 初始化完成，环缓冲 (ch={self.n_chans}, len={self.max_len}), "
              f"滤波器 sos 预计算完成")

    # ------------------------------------------------------------------
    def feed(self, sample):
        """Feed one sample (shape: (channels,)) and return a decision if ready.

        Returns (decision, confidence, current_time_sec) or (None, 0.0, t).
        """
        # -- ring-buffer write --
        self._buf[:, self._write] = sample
        self._write = (self._write + 1) % self.max_len
        if self._total < self.max_len:
            self._total += 1
        else:
            # buffer full → advance start to keep window at max_len
            self._start = (self._start + 1) % self.max_len

        L = self._total

        if L < self.min_len:
            return None, 0.0, L / self.sample_rate

        # Only check at step boundaries
        if L % self.step != 0:
            return None, 0.0, L / self.sample_rate

        # ---- select the largest model ≤ L ----
        model_len = None
        for wl in self.model_lengths:
            if wl <= L:
                model_len = wl
            else:
                break

        if model_len is None:
            return None, 0.0, L / self.sample_rate

        # ---- extract and preprocess ----
        window = self._extract(model_len)
        window = self._preprocess(window)

        # ---- predict ----
        scores = self.models[model_len].transform(window[np.newaxis, ...])[0]

        top2 = np.partition(scores, -2)[-2:]
        margin = top2.max() - top2.min()
        max_score = np.max(scores)
        decision = np.argmax(scores)

        # Cache the preprocessed window for possible forced-output reuse
        self._cached_window = window
        self._cached_model_len = model_len

        # ---- early-stop check ----
        if margin > self.margin_th and max_score > self.max_th:
            self.history.append(decision)
            if len(self.history) == self.cons_req and len(set(self.history)) == 1:
                return decision, max_score, L / self.sample_rate
        else:
            self.history.clear()

        # ---- forced output at max_len ----
        if L >= self.max_len:
            # Reuse cached window if it was for the max model, else recompute
            if self._cached_model_len == self.max_len and self._cached_window is not None:
                scores_force = self.models[self.max_len].transform(
                    self._cached_window[np.newaxis, ...])[0]
            else:
                window_force = self._extract(self.max_len)
                window_force = self._preprocess(window_force)
                scores_force = self.models[self.max_len].transform(
                    window_force[np.newaxis, ...])[0]
            decision_force = np.argmax(scores_force)
            conf_force = np.max(scores_force)
            return decision_force, conf_force, self.max_len / self.sample_rate

        return None, 0.0, L / self.sample_rate

    # ------------------------------------------------------------------
    def slide(self, n=None):
        """Discard the oldest *n* samples (default: step).

        Used in REALTIME continuous decoding to keep the buffer from growing
        unboundedly while preserving recent history.
        """
        if n is None:
            n = self.step
        if n >= self._total:
            self.reset()
            return
        self._start = (self._start + n) % self.max_len
        self._total -= n
        self._cached_window = None
        self._cached_model_len = None

    # ------------------------------------------------------------------
    def reset(self):
        """Full reset for a new trial."""
        self._buf.fill(0.0)
        self._start = 0
        self._write = 0
        self._total = 0
        self.history.clear()
        self._cached_window = None
        self._cached_model_len = None

    # ------------------------------------------------------------------
    def _extract(self, model_len):
        """Copy *model_len* samples from the ring buffer as (channels, samples)."""
        if self._start + model_len <= self.max_len:
            return self._buf[:, self._start:self._start + model_len].copy()
        else:
            first = self.max_len - self._start
            return np.hstack([
                self._buf[:, self._start:],
                self._buf[:, :model_len - first]
            ])

    # ------------------------------------------------------------------
    def _preprocess(self, window):
        """De-mean + narrow-band enhancement using pre-computed SOS filter."""
        window = window - np.mean(window, axis=1, keepdims=True)
        filtered = sosfilt(self._sos, window, axis=-1)
        return window + 0.5 * filtered
