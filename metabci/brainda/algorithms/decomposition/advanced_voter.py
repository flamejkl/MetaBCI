# -*- coding: utf-8 -*-
# License: MIT License
"""
Advanced Voter for online BCI decision smoothing.

Provides exponential-decay confidence accumulation with consecutive-confirmation
locking to reduce spurious command flicker in continuous decoding.

Reference
---------
.. code-block:: python

    from metabci.brainda.algorithms.decomposition import AdvancedVoter

    voter = AdvancedVoter(decay=0.8, lock_frames=3, lock_duration=0.5, threshold=0.5)
    for prob in classifier_stream:          # prob shape (4,)
        cmd, conf = voter.update(prob)
        if cmd is not None:
            execute(cmd)
"""
import time
import numpy as np


class AdvancedVoter:
    """Confidence accumulator with exponential decay and lock-out.

    Parameters
    ----------
    decay : float
        EMA decay factor for accumulated probabilities (0 < decay < 1).
    lock_frames : int
        Number of consecutive consistent frames required to lock a decision.
    lock_duration : float
        Duration (seconds) to hold the locked decision before allowing a new one.
    threshold : float
        Minimum accumulated confidence to emit a decision.
    """

    def __init__(self, decay=0.8, lock_frames=3, lock_duration=0.5, threshold=0.5):
        self.decay = decay
        self.lock_frames = lock_frames
        self.lock_duration = lock_duration
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset voter state (call at the start of each trial / session)."""
        self.accumulated = np.zeros(4)
        self.consecutive = 0
        self.locked_dir = None
        self.lock_until = 0.0
        self.last_dir = None

    def update(self, current_prob, timestamp=None):
        """Ingest a probability vector and return a decision if ready.

        Parameters
        ----------
        current_prob : array-like, shape (n_classes,)
        timestamp : float or None
            Current time in seconds (default: ``time.time()``).

        Returns
        -------
        direction_idx : int or None
        confidence : float or None
        """
        if timestamp is None:
            timestamp = time.time()

        self.accumulated = (self.decay * self.accumulated +
                            (1 - self.decay) * np.array(current_prob))
        best_idx = np.argmax(self.accumulated)
        best_conf = self.accumulated[best_idx]

        if self.locked_dir is not None and timestamp < self.lock_until:
            return self.locked_dir, best_conf

        if best_conf >= self.threshold:
            if self.last_dir == best_idx:
                self.consecutive += 1
            else:
                self.consecutive = 1
            self.last_dir = best_idx

            if self.consecutive >= self.lock_frames:
                self.locked_dir = best_idx
                self.lock_until = timestamp + self.lock_duration
                self.consecutive = 0
            return best_idx, best_conf
        else:
            self.consecutive = 0
            self.last_dir = None
            return None, best_conf
