# -*- coding: utf-8 -*-
# License: MIT License
"""
SSVEP online decoding worker based on brainflow ProcessWorker.

Provides SSVEPWorker — a brainflow ProcessWorker subclass that runs GrowingWindowDecoder
in a subprocess for real-time SSVEP classification, keeping the main process
responsive for WebSocket I/O and stimulus timing.

Reference
---------
.. code-block:: python

    from metabci.brainflow.ssvep_worker import SSVEPWorker

    worker = SSVEPWorker(
        model_paths={125: "model_125.pkl", ...},
        occipital_indices=[2,3,4,5,6,7,8,9],
    )
    worker.pre()      # init decoder
    worker.start()     # launch subprocess
    worker.put(trial)  # feed (n_samples, n_channels) trial
    # subprocess runs consume(), posts result to worker.result_queue
"""
import time
import queue
import threading
import numpy as np
from typing import Optional, Dict, Tuple

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.growing_window import (
    GrowingWindowDecoder,
)


class SSVEPWorker(ProcessWorker):
    """brainflow ProcessWorker for real-time SSVEP decoding.

    Runs GrowingWindowDecoder in a subprocess so that blocking model inference
    does not stall WebSocket I/O or stimulus timing in the main process.

    Parameters
    ----------
    model_paths : dict
        {window_samples: .pkl path}.
    occipital_indices : list of int
        Channel indices for occipital electrodes.
    sample_rate : int
        EEG sampling rate in Hz.
    timeout : float
        queue polling timeout in seconds.
    """

    def __init__(
        self,
        model_paths: Optional[Dict[int, str]] = None,
        occipital_indices: Optional[list] = None,
        sample_rate: int = 250,
        timeout: float = 1e-3,
        name: Optional[str] = None,
    ):
        super().__init__(timeout=timeout, name=name)
        self.model_paths = model_paths
        self.occipital_indices = occipital_indices or [2, 3, 4, 5, 6, 7, 8, 9]
        self.sample_rate = sample_rate
        self._decoder: Optional[GrowingWindowDecoder] = None

        # Thread-safe result queue so main process can poll decisions
        self.result_queue: queue.Queue = queue.Queue()

    # ==================================================================
    #  ProcessWorker interface
    # ==================================================================

    def pre(self):
        """Initialise decoder in subprocess before real-time loop."""
        self._decoder = GrowingWindowDecoder(
            model_paths=self.model_paths,
            occipital_indices=self.occipital_indices,
            sample_rate=self.sample_rate,
        )

    def consume(self, data: np.ndarray):
        """Feed one trial worth of EEG samples.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_channels)
            A full trial (typically 500 samples × 8 channels).
        """
        if self._decoder is None:
            return

        self._decoder.reset()
        for sample in data:
            decision, confidence, t = self._decoder.feed(sample)
            if decision is not None:
                self.result_queue.put({
                    "decision": decision,
                    "confidence": confidence,
                    "decision_time": t,
                    "scores": getattr(self._decoder, '_last_scores', None),
                })
                break
        else:
            # Trial exhausted without decision (should not happen)
            # Force with last state
            self.result_queue.put({
                "decision": -1,
                "confidence": 0.0,
                "decision_time": 2.0,
                "scores": None,
            })

    def post(self):
        """Cleanup."""
        if self._decoder:
            self._decoder.reset_normaliser()

    def calibrate(self, samples: np.ndarray):
        """Seed online normaliser with resting EEG."""
        if self._decoder:
            self._decoder.calibrate(samples)

    # ==================================================================
    #  Convenience (single-process fallback)
    # ==================================================================

    def run_trial(self, data: np.ndarray) -> dict:
        """Decode one trial synchronously (no subprocess needed).

        Returns dict with keys: decision, confidence, decision_time, scores.
        """
        if self._decoder is None:
            self.pre()
        self._decoder.reset()
        for sample in data:
            d, conf, t = self._decoder.feed(sample)
            if d is not None:
                return {
                    "decision": d, "confidence": conf,
                    "decision_time": t,
                    "scores": getattr(self._decoder, '_last_scores', None),
                }
        return {"decision": -1, "confidence": 0.0, "decision_time": 2.0, "scores": None}
