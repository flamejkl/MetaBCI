# -*- coding: utf-8 -*-
# License: MIT License
"""
SSVEP online decoding worker based on brainflow ProcessWorker.

Provides SSVEPWorker — a brainflow ProcessWorker subclass that receives
EEG data through the multiprocessing queue and runs GrowingWindowDecoder
for real-time SSVEP classification.

Also provides DataAcquisitionWrapper — a thin wrapper that demonstrates
brainflow integration for EEG data collection with Neuracle hardware.

Reference
---------
.. code-block:: python

    from metabci.brainflow.ssvep_worker import SSVEPWorker

    worker = SSVEPWorker(
        model_paths={...},
        occipital_indices=[2,3,4,5,6,7,8,9],
    )
    worker.start()
    worker.put(eeg_trial)        # from main process
    # worker runs consume() in subprocess → emits decisions
"""
import time
import numpy as np
from typing import Optional, Dict, Tuple

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.growing_window import (
    GrowingWindowDecoder,
)


class SSVEPWorker(ProcessWorker):
    """brainflow ProcessWorker for online SSVEP decoding.

    Parameters
    ----------
    model_paths : dict
        {window_length_samples: model_path} for GrowingWindowDecoder.
    occipital_indices : list of int
        Channel indices for occipital electrodes.
    sample_rate : int
        EEG sampling rate in Hz.
    decision_callback : callable or None
        Called as ``callback(decision_idx, confidence)`` when a decision is made.
    """

    def __init__(
        self,
        model_paths: Optional[Dict[int, str]] = None,
        occipital_indices: Optional[list] = None,
        sample_rate: int = 250,
        decision_callback=None,
        timeout: float = 1e-3,
        name: Optional[str] = None,
    ):
        super().__init__(timeout=timeout, name=name)
        self.model_paths = model_paths
        self.occipital_indices = occipital_indices or [2, 3, 4, 5, 6, 7, 8, 9]
        self.sample_rate = sample_rate
        self.decision_callback = decision_callback
        self.decoder: Optional[GrowingWindowDecoder] = None
        self._n_samples = 0

    # ---- ProcessWorker interface ----
    def pre(self):
        """Offline preparation — instantiate the decoder."""
        self.decoder = GrowingWindowDecoder(
            model_paths=self.model_paths,
            sample_rate=self.sample_rate,
            occipital_indices=self.occipital_indices,
        )
        self._n_samples = 0

    def consume(self, data: np.ndarray):
        """Online processing — feed EEG samples through the decoder.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_channels + 1)
            Single trial of online data.  The last channel is the trigger
            channel (ignored for decoding).
        """
        if self.decoder is None:
            return

        # Discard trigger channel if present
        eeg = data[:, :14] if data.shape[1] >= 14 else data
        # Select occipital channels
        eeg = eeg[:, self.occipital_indices]

        for sample in eeg:
            decision, confidence, t = self.decoder.feed(sample)
            if decision is not None:
                if self.decision_callback:
                    self.decision_callback(decision, confidence)
                self.decoder.reset()
                break

    def post(self):
        """Cleanup after stopping."""
        if self.decoder:
            self.decoder.reset_normaliser()
        self._n_samples = 0

    def calibrate(self, samples: np.ndarray):
        """Seed the decoder's online normaliser with background EEG."""
        if self.decoder:
            self.decoder.calibrate(samples)
