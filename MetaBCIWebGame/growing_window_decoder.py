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
    """Growing-window SSVEP decoder with dynamic stopping, online normalization,
    and adaptive thresholding to mitigate experiment→game distribution shift.

    Key anti-shift mechanisms
    -------------------------
    1. **Online EMA z-score** – tracks per-channel running mean/std to normalise
       slow impedance / conductance drift between calibration and gameplay.
    2. **Adaptive threshold** – lowers the stopping margin when recent decisions
       are consistently confident (game context matches training), and raises it
       when the stream becomes noisy.
    3. **Short calibration** – ``calibrate(samples)`` can ingest a few seconds of
       resting-state EEG to seed the normaliser before the first trial.
    """

    def __init__(self, model_paths=None,
                 enable_online_norm=True,
                 enable_adaptive_threshold=True):
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
        self._start = 0
        self._write = 0
        self._total = 0

        # ---- consecutive-decision history ----
        self.history = deque(maxlen=self.cons_req)

        # ---- pre-compute filter coefficients ONCE ----
        fs = self.sample_rate
        self._sos = butter(4, [7, 18], btype='bandpass', fs=fs, output='sos')

        # ---- cache for forced-output reuse ----
        self._cached_window = None
        self._cached_model_len = None

        # ==================================================================
        #  Layer 1 – Online EMA z-score normalisation
        # ==================================================================
        self._online_norm = enable_online_norm
        # Running EMA statistics (per-channel)
        self._ema_mean = np.zeros(self.n_chans)       # μ
        self._ema_var = np.ones(self.n_chans)          # σ²
        self._ema_decay = 0.995                        # ~5 s time-constant at 250 Hz
        self._ema_warmup = int(2.0 * self.sample_rate) # 2 s warm-up
        self._ema_count = 0
        # Floor for std to avoid divide-by-zero
        self._eps = 1e-8

        # ==================================================================
        #  Layer 3 – Adaptive threshold
        # ==================================================================
        self._adaptive = enable_adaptive_threshold
        # Rolling window of recent (margin, max_score, success) tuples
        self._recent_quality = deque(maxlen=20)
        self._adaptive_margin_min = 0.15   # floor – margin will never go below this
        self._adaptive_margin_max = 0.50   # ceiling
        self._adaptive_maxscore_min = 0.50
        self._adaptive_maxscore_max = 0.85

        print(f"[GWDecoder] 初始化完成，环缓冲 (ch={self.n_chans}, len={self.max_len})")
        print(f"[GWDecoder] 在线归一化: {'开' if self._online_norm else '关'}, "
              f"自适应阈值: {'开' if self._adaptive else '关'}, "
              f"滤波器 sos 预计算完成")

    # ======================================================================
    #  Public API
    # ======================================================================

    def calibrate(self, samples):
        """Ingest *samples* (n_samples, n_channels) of resting / background EEG
        to seed the online normaliser.  Call once after the game starts but
        before the first trial (e.g. 2–3 s of data)."""
        if not self._online_norm:
            return
        if samples.ndim != 2 or samples.shape[1] != self.n_chans:
            samples = samples.T
        for s in samples:
            self._update_ema(s)

    def feed(self, sample):
        """Feed one sample (shape: (channels,)) and return a decision if ready.

        Returns (decision, confidence, current_time_sec) or (None, 0.0, t).
        """
        # -- online normalisation --
        if self._online_norm:
            self._update_ema(sample)
            sample = self._normalise(sample)

        # -- ring-buffer write --
        self._buf[:, self._write] = sample
        self._write = (self._write + 1) % self.max_len
        if self._total < self.max_len:
            self._total += 1
        else:
            self._start = (self._start + 1) % self.max_len

        L = self._total

        if L < self.min_len:
            return None, 0.0, L / self.sample_rate

        if L % self.step != 0:
            return None, 0.0, L / self.sample_rate

        # ---- select the largest model ≤ L ----
        model_len = self._best_model_len(L)
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

        self._cached_window = window
        self._cached_model_len = model_len

        # ---- dynamic thresholds ----
        eff_margin_th, eff_max_th = self._effective_thresholds()

        # ---- early-stop check ----
        if margin > eff_margin_th and max_score > eff_max_th:
            self.history.append(decision)
            if len(self.history) == self.cons_req and len(set(self.history)) == 1:
                if self._adaptive:
                    self._recent_quality.append((margin, max_score, True))
                return decision, max_score, L / self.sample_rate
        else:
            self.history.clear()

        # ---- forced output at max_len ----
        if L >= self.max_len:
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
            if self._adaptive:
                # forced output → lower quality signal
                self._recent_quality.append((0.0, conf_force, False))
            return decision_force, conf_force, self.max_len / self.sample_rate

        return None, 0.0, L / self.sample_rate

    def slide(self, n=None):
        """Discard the oldest *n* samples (default: step)."""
        if n is None:
            n = self.step
        if n >= self._total:
            self.reset()
            return
        self._start = (self._start + n) % self.max_len
        self._total -= n
        self._cached_window = None
        self._cached_model_len = None

    def reset(self):
        """Full reset for a new trial (does NOT reset EMA normaliser)."""
        self._buf.fill(0.0)
        self._start = 0
        self._write = 0
        self._total = 0
        self.history.clear()
        self._cached_window = None
        self._cached_model_len = None

    def reset_normaliser(self):
        """Reset the online EMA normaliser (e.g. on session restart)."""
        self._ema_mean = np.zeros(self.n_chans)
        self._ema_var = np.ones(self.n_chans)
        self._ema_count = 0

    def get_normaliser_state(self):
        """Return current EMA mean/std for debugging / logging."""
        return {
            'mean': self._ema_mean.copy(),
            'std': np.sqrt(np.maximum(self._ema_var, self._eps)),
            'count': self._ema_count
        }

    # ======================================================================
    #  Internal helpers
    # ======================================================================

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

    def _preprocess(self, window):
        """De-mean + narrow-band enhancement using pre-computed SOS filter."""
        window = window - np.mean(window, axis=1, keepdims=True)
        filtered = sosfilt(self._sos, window, axis=-1)
        return window + 0.5 * filtered

    def _best_model_len(self, L):
        for wl in self.model_lengths:
            if wl <= L:
                continue
            idx = self.model_lengths.index(wl) - 1
            return self.model_lengths[idx] if idx >= 0 else None
        return self.model_lengths[-1] if self.model_lengths else None

    # ------------------------------------------------------------------
    #  Layer 1 – online EMA z-score
    # ------------------------------------------------------------------
    def _update_ema(self, sample):
        """Update running per-channel mean and variance via exponential smoothing."""
        self._ema_count += 1
        if self._ema_count == 1:
            self._ema_mean = sample.astype(np.float64).copy()
            self._ema_var = np.ones(self.n_chans)  # start with unit variance
            return
        alpha = 1.0 - self._ema_decay
        self._ema_mean = (1 - alpha) * self._ema_mean + alpha * sample
        delta = sample - self._ema_mean
        self._ema_var = (1 - alpha) * self._ema_var + alpha * (delta * delta)

    def _normalise(self, sample):
        """Apply z-score normalisation: (x - μ) / σ.
        During warm-up, only subtract mean (no scaling) to avoid amplifying noise."""
        std = np.sqrt(np.maximum(self._ema_var, self._eps))
        if self._ema_count < self._ema_warmup:
            # Warm-up: only de-mean, don't scale
            return sample - self._ema_mean
        return (sample - self._ema_mean) / std

    # ------------------------------------------------------------------
    #  Layer 3 – adaptive threshold
    # ------------------------------------------------------------------
    def _effective_thresholds(self):
        """Return (margin_th, max_th) possibly adapted from recent quality."""
        if not self._adaptive or len(self._recent_quality) < 5:
            return self.margin_th, self.max_th

        # Fraction of recent decisions that were high-quality (early stop)
        early_stops = sum(1 for m, s, ok in self._recent_quality if ok)
        frac = early_stops / len(self._recent_quality)

        # Smooth interpolation: high quality → lower thresholds (easier to decide)
        #                      low quality  → higher thresholds (more conservative)
        margin = self._adaptive_margin_min + \
                 (1.0 - frac) * (self._adaptive_margin_max - self._adaptive_margin_min)
        max_s = self._adaptive_maxscore_min + \
                (1.0 - frac) * (self._adaptive_maxscore_max - self._adaptive_maxscore_min)

        # Clamp to config bounds when quality is very high
        margin = np.clip(margin, self._adaptive_margin_min,
                         self.margin_th)
        max_s = np.clip(max_s, self._adaptive_maxscore_min,
                        self.max_th)

        return margin, max_s
