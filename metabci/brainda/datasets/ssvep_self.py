# -*- coding: utf-8 -*-
# License: MIT License
"""
Self-collected SSVEP dataset for MetaBCIWebGame.

Four-class SSVEP (up/down/left/right at 8.25 / 11.0 / 13.75 / 16.5 Hz),
14-channel Neuracle acquisition, 250 Hz sampling rate, 2-second trials.

Implements the BaseDataset protocol for MetaBCI compatibility and supports
multi-subject directory layouts.

Directory layout (flat, single subject)::

    data_root/
        1/   hw_trial_0000_offset000.npy ...
        2/   ...
        3/   ...
        4/   ...

Directory layout (multi-subject)::

    data_root/
        sub-01/
            1/   ...
            2/   ...
            3/   ...
            4/   ...
        sub-02/
            ...

Reference
---------
.. code-block:: python

    from metabci.brainda.datasets import SelfSSVEP

    ds = SelfSSVEP(data_root="data_self")
    X, y = ds.get_data()                        # numpy arrays
    raw = ds.get_mne_raw(subject="sub-01")      # MNE Raw objects

    # Multi-subject
    ds2 = SelfSSVEP(data_root="data_multi", subject_id="sub-02")
"""
import os
import glob
import re
import numpy as np
from typing import Optional, List, Union, Dict, Tuple

_OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]
_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'O1', 'O2', 'Oz',
    'PO3', 'PO4', 'PO5', 'PO6', 'POz',
    'P3', 'P4', 'P7', 'P8',
]
_OCCIPITAL_NAMES = [_CHANNEL_NAMES[i] for i in _OCCIPITAL_INDICES]


class SelfSSVEP:
    """Self-collected SSVEP dataset — MetaBCI compatible.

    Parameters
    ----------
    data_root : str
        Root folder containing label sub-directories ``1/``…``4/``.
    occipital_indices : list of int, optional
        Channel indices for occipital electrodes.
    offset_only : bool
        Only load ``offset000`` files.
    subject_id : str or None
        If given, load from ``data_root/<subject_id>/`` sub-folder
        (multi-subject layout).  If None, load directly from *data_root*
        (single-subject flat layout).
    """

    # ---- MetaBCI BaseDataset compatible metadata ----
    DATASET_CODE = "SelfSSVEP"
    PARADIGM = "ssvep"
    SRATE = 250
    N_CLASSES = 4
    STIM_FREQS = [8.25, 11.0, 13.75, 16.5]
    STIM_LABELS = ['up', 'down', 'left', 'right']
    EVENTS = {
        'up':    (0, (0.0, 2.0)),
        'down':  (1, (0.0, 2.0)),
        'left':  (2, (0.0, 2.0)),
        'right': (3, (0.0, 2.0)),
    }
    CHANNELS = _CHANNEL_NAMES
    OCCIPITAL_CHANNELS = _OCCIPITAL_NAMES

    def __init__(self, data_root, occipital_indices=None, offset_only=True,
                 subject_id: Optional[str] = None):
        self.data_root = data_root
        self.occipital_indices = (list(occipital_indices) if occipital_indices is not None
                                  else list(_OCCIPITAL_INDICES))
        self.offset_only = offset_only
        self.subject_id = subject_id

        # Resolve actual load path
        if subject_id:
            self._load_root = os.path.join(data_root, subject_id)
        else:
            self._load_root = data_root

        self._srate = self.SRATE
        self._n_classes = self.N_CLASSES
        self._stim_freqs = list(self.STIM_FREQS)
        self._stim_labels = list(self.STIM_LABELS)
        self.trials_by_label: Dict[int, List[Tuple[np.ndarray, int, str]]] = {
            0: [], 1: [], 2: [], 3: []
        }
        self._subject_ids: List[str] = []
        self._load()

    # ---- MetaBCI dataset properties ----
    @property
    def dataset_code(self) -> str:
        return self.DATASET_CODE

    @property
    def srate(self) -> int:
        return self._srate

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def stim_freqs(self) -> List[float]:
        return self._stim_freqs

    @property
    def stim_labels(self) -> List[str]:
        return self._stim_labels

    @property
    def subjects(self) -> List[str]:
        """List of available subject IDs (multi-subject mode) or [\'default\']."""
        if self._subject_ids:
            return self._subject_ids
        return ['default']

    @property
    def channels(self) -> List[str]:
        return list(_CHANNEL_NAMES)

    @property
    def events(self) -> Dict[str, Tuple[int, Tuple[float, float]]]:
        return dict(self.EVENTS)

    @property
    def paradigm(self) -> str:
        return self.PARADIGM

    # ---- data loading ----
    def _load(self):
        self._scan_subjects()
        root = self._load_root
        for label in range(4):
            folder = os.path.join(root, str(label + 1))
            if not os.path.isdir(folder):
                continue
            for f in glob.glob(os.path.join(folder, "*.npy")):
                if "hw_trial_0000.npy" in f:
                    continue
                if self.offset_only and "offset000" not in f:
                    continue
                data = np.load(f)
                data = data[self.occipital_indices, :]
                self.trials_by_label[label].append(
                    (data, label, os.path.basename(f)))
        total = sum(len(v) for v in self.trials_by_label.values())
        if total == 0:
            raise RuntimeError(
                f"No valid .npy trials found under {self._load_root}")

    def _scan_subjects(self):
        """Detect sub-* directories for multi-subject layout."""
        if self.subject_id is not None:
            return  # explicit subject, no scan needed
        candidates = []
        if os.path.isdir(self.data_root):
            for name in os.listdir(self.data_root):
                sub_dir = os.path.join(self.data_root, name)
                if os.path.isdir(sub_dir) and re.match(r'sub-', name):
                    # Verify it has label directories
                    if any(os.path.isdir(os.path.join(sub_dir, str(l)))
                           for l in range(1, 5)):
                        candidates.append(name)
        self._subject_ids = sorted(candidates)

    # ---- numpy API (for sklearn / game engine) ----
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) — (n_trials, n_channels, n_samples), (n_trials,)."""
        X_list, y_list = [], []
        for label in range(4):
            for data, lbl, _ in self.trials_by_label[label]:
                X_list.append(data)
                y_list.append(lbl)
        if not X_list:
            raise RuntimeError("No data loaded")
        return np.stack(X_list, axis=0), np.array(y_list)

    def get_random_trial(self, label: int) -> Tuple[np.ndarray, int, str]:
        """Return (data, label, filename) for a random trial of *label*."""
        trials = self.trials_by_label.get(label, [])
        if not trials:
            raise ValueError(f"No trials for label {label}")
        return trials[np.random.randint(0, len(trials))]

    def random_generator(self, label: int):
        """Infinite generator of random trials for *label*."""
        trials = self.trials_by_label.get(label, [])
        if not trials:
            raise ValueError(f"No trials for label {label}")
        n = len(trials)
        while True:
            yield trials[np.random.randint(0, n)]

    # ---- MNE Raw API (MetaBCI BaseDataset protocol) ----
    def get_mne_raw(self, subject: str = 'default'):
        """Build MNE RawArray for *subject*.

        Requires ``mne`` to be installed. Returns a dict ``{'session_0':
        {'run_0': mne.io.RawArray}}`` compatible with BaseDataset protocol.
        """
        try:
            import mne
        except ImportError:
            raise ImportError("mne is required for get_mne_raw()")
        X, y = self.get_data()
        info = mne.create_info(
            ch_names=self.OCCIPITAL_CHANNELS,
            sfreq=self._srate,
            ch_types='eeg',
        )
        # Stack all trials horizontally as one continuous recording
        data = X.reshape(-1, X.shape[2])  # (n_trials*channels, samples) → hstack
        data = np.hstack([X[i] for i in range(X.shape[0])])  # (ch, total_samples)
        raw = mne.io.RawArray(data, info)
        return {'session_0': {'run_0': raw}}

    def _get_single_subject_data(self, subject: str = 'default'):
        """MetaBCI BaseDataset protocol — returns MNE Raw dict."""
        # If a different subject than loaded, create a new instance
        if subject != (self.subject_id or 'default'):
            ds = SelfSSVEP(
                data_root=self.data_root,
                occipital_indices=self.occipital_indices,
                offset_only=self.offset_only,
                subject_id=subject if subject != 'default' else None,
            )
            return ds.get_mne_raw(subject)
        return self.get_mne_raw(subject)

    def __repr__(self):
        total = sum(len(v) for v in self.trials_by_label.values())
        subj = self.subject_id or 'default'
        return (f"SelfSSVEP(subject={subj}, trials={total}, "
                f"classes={self._n_classes}, srate={self._srate})")
