# -*- coding: utf-8 -*-
# License: MIT License
"""
Growing Window SSVEP Decoder with Dynamic Stopping.

Provides GrowingWindowDecoder — an online SSVEP decoder that accumulates
samples in a pre-allocated ring buffer, checks multiple models (125 / 250 /
375 / 500 samples) at step intervals, and stops early when margin + confidence
criteria are met.  Includes online EMA de-meaning and adaptive thresholding to
reduce experiment→game distribution shift.

Reference
---------
.. code-block:: python

    from metabci.brainda.algorithms.decomposition import GrowingWindowDecoder

    dec = GrowingWindowDecoder(model_paths={
        125: "model_125.pkl",
        250: "model_250.pkl",
        375: "model_375.pkl",
        500: "self_ssvep_model.pkl",
    })
    for sample in eeg_stream:                # sample shape (n_channels,)
        decision, conf, t = dec.feed(sample)
        if decision is not None:
            print(f"Decided {decision} at {t:.2f}s")
            dec.slide()                      # for continuous decoding
"""
import numpy as np
import joblib
from collections import deque
from scipy.signal import butter, sosfilt

# Default paths mirror the MetaBCIWebGame convention; override via __init__
_DEFAULT_MODEL_PATHS = {
    125: "model_125.pkl",
    250: "model_250.pkl",
    375: "model_375.pkl",
    500: "self_ssvep_model.pkl",
}
_DEFAULT_SAMPLE_RATE = 250
_DEFAULT_OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]


class GrowingWindowDecoder:
    """Growing-window SSVEP decoder with dynamic stopping."""

    def __init__(
        self,
        model_paths=None,
        sample_rate=_DEFAULT_SAMPLE_RATE,
        occipital_indices=None,
        step=25,
        min_len=125,
        max_len=500,
        margin_th=0.35,
        max_th=0.75,
        cons_req=1,
        filter_band=(7, 22),
        filter_order=4,
        enable_online_norm=True,
        enable_adaptive_threshold=True,
    ):
        if model_paths is None:
            model_paths = _DEFAULT_MODEL_PATHS
        if occipital_indices is None:
            occipital_indices = list(_DEFAULT_OCCIPITAL_INDICES)

        # ---- load models ----
        self.models = {}
        for L, path in model_paths.items():
            self.models[L] = joblib.load(path)

        # ---- config ----
        self.step = step
        self.margin_th = margin_th
        self.max_th = max_th
        self.cons_req = cons_req
        self.min_len = min_len
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.n_chans = len(occipital_indices)
        self.model_lengths = sorted(self.models.keys())

        # ---- ring buffer ----
        self._buf = np.zeros((self.n_chans, self.max_len), dtype=np.float64)
        self._start = 0
        self._write = 0
        self._total = 0

        # ---- consecutive-decision history ----
        self.history = deque(maxlen=self.cons_req)

        # ---- pre-compute filter ----
        fs = self.sample_rate
        self._sos = butter(filter_order, filter_band, btype='bandpass',
                           fs=fs, output='sos')

        # ---- cache ----
        self._cached_window = None
        self._cached_model_len = None

        # ---- online EMA normalisation ----
        self._online_norm = enable_online_norm
        self._ema_mean = np.zeros(self.n_chans)
        self._ema_var = np.ones(self.n_chans)
        self._ema_decay = 0.995
        self._ema_warmup = int(2.0 * self.sample_rate)
        self._ema_count = 0
        self._eps = 1e-8

        # ---- adaptive threshold ----
        self._adaptive = enable_adaptive_threshold
        self._recent_quality = deque(maxlen=20)
        self._adaptive_margin_min = 0.15
        self._adaptive_margin_max = 0.50
        self._adaptive_maxscore_min = 0.50
        self._adaptive_maxscore_max = 0.85

    # ------------------------------------------------------------------
    def calibrate(self, samples):
        """Seed the online normaliser with background EEG.

        Parameters
        ----------
        samples : ndarray (n_samples, n_channels) or (n_channels, n_samples)
        """
        if not self._online_norm:
            return
        if samples.ndim != 2 or samples.shape[1] != self.n_chans:
            samples = samples.T
        for s in samples:
            self._update_ema(s)

    def feed(self, sample):
        """Feed one sample (shape: (channels,)) → (decision, confidence, t)."""
        if self._online_norm:
            self._update_ema(sample)
            sample = self._normalise(sample)

        self._buf[:, self._write] = sample
        self._write = (self._write + 1) % self.max_len
        if self._total < self.max_len:
            self._total += 1
        else:
            self._start = (self._start + 1) % self.max_len

        L = self._total
        if L < self.min_len:
            return None, 0.0, L / self.sample_rate
        # Only evaluate at model-length boundaries (125, 250, 375, 500),
        # skipping redundant intermediate checks that re-run the same model
        if L not in self.model_lengths and L < self.max_len:
            return None, 0.0, L / self.sample_rate

        model_len = L if L in self.model_lengths else self.max_len
        if model_len is None:
            return None, 0.0, L / self.sample_rate

        window = self._extract(model_len)
        window = self._preprocess(window)

        scores = self.models[model_len].transform(window[np.newaxis, ...])[0]
        top2 = np.partition(scores, -2)[-2:]
        margin = top2.max() - top2.min()
        max_score = np.max(scores)
        decision = np.argmax(scores)

        self._cached_window = window
        self._cached_model_len = model_len

        eff_margin, eff_max = self._effective_thresholds()

        if margin > eff_margin and max_score > eff_max:
            self.history.append(decision)
            if len(self.history) == self.cons_req and len(set(self.history)) == 1:
                if self._adaptive:
                    self._recent_quality.append((margin, max_score, True))
                return decision, max_score, L / self.sample_rate
        else:
            self.history.clear()

        if L >= self.max_len:
            if self._cached_model_len == self.max_len and self._cached_window is not None:
                scores_force = self.models[self.max_len].transform(
                    self._cached_window[np.newaxis, ...])[0]
            else:
                wf = self._extract(self.max_len)
                wf = self._preprocess(wf)
                scores_force = self.models[self.max_len].transform(
                    wf[np.newaxis, ...])[0]
            df = np.argmax(scores_force)
            cf = np.max(scores_force)
            if self._adaptive:
                self._recent_quality.append((0.0, cf, False))
            return df, cf, self.max_len / self.sample_rate

        return None, 0.0, L / self.sample_rate

    def slide(self, n=None):
        """Discard oldest *n* samples (default: step)."""
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
        """Full reset for a new trial (preserves EMA normaliser)."""
        self._buf.fill(0.0)
        self._start = 0
        self._write = 0
        self._total = 0
        self.history.clear()
        self._cached_window = None
        self._cached_model_len = None

    def reset_normaliser(self):
        """Reset the online EMA normaliser."""
        self._ema_mean = np.zeros(self.n_chans)
        self._ema_var = np.ones(self.n_chans)
        self._ema_count = 0

    # ------------------------------------------------------------------
    def _extract(self, model_len):
        if self._start + model_len <= self.max_len:
            return self._buf[:, self._start:self._start + model_len].copy()
        first = self.max_len - self._start
        return np.hstack([
            self._buf[:, self._start:],
            self._buf[:, :model_len - first],
        ])

    def _preprocess(self, window):
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

    def _update_ema(self, sample):
        self._ema_count += 1
        if self._ema_count == 1:
            self._ema_mean = sample.astype(np.float64).copy()
            self._ema_var = np.ones(self.n_chans)
            return
        alpha = 1.0 - self._ema_decay
        self._ema_mean = (1 - alpha) * self._ema_mean + alpha * sample
        delta = sample - self._ema_mean
        self._ema_var = (1 - alpha) * self._ema_var + alpha * (delta * delta)

    def _normalise(self, sample):
        return sample - self._ema_mean

    def _effective_thresholds(self):
        if not self._adaptive or len(self._recent_quality) < 5:
            return self.margin_th, self.max_th
        early_stops = sum(1 for _, _, ok in self._recent_quality if ok)
        frac = early_stops / len(self._recent_quality)
        margin = self._adaptive_margin_min + (1.0 - frac) * (
            self._adaptive_margin_max - self._adaptive_margin_min)
        max_s = self._adaptive_maxscore_min + (1.0 - frac) * (
            self._adaptive_maxscore_max - self._adaptive_maxscore_min)
        return (np.clip(margin, self._adaptive_margin_min, self.margin_th),
                np.clip(max_s, self._adaptive_maxscore_min, self.max_th))
