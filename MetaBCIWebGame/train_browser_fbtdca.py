# -*- coding: utf-8 -*-
"""Leakage-free FBTDCA training for browser-collected SSVEP trials."""

import argparse
import json
from pathlib import Path
from copy import deepcopy

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from metabci.brainda.algorithms.decomposition.base import (
    generate_cca_references,
    generate_filterbank,
)
from metabci.brainda.algorithms.decomposition.tdca import FBTDCA


SAMPLE_RATE = 250
TARGET_FREQS = [8.25, 11.0, 13.75, 16.5]


def load_session(session_path, window_samples=500):
    """Load four class directories in stable filename order."""
    root = Path(session_path)
    trials, labels, trial_ids = [], [], []
    for label in range(4):
        class_dir = root / str(label + 1)
        if not class_dir.is_dir():
            raise ValueError(f"Missing class directory: {class_dir}")
        files = sorted(class_dir.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy trials in: {class_dir}")
        for path in files:
            data = np.load(path, allow_pickle=False)
            if data.ndim != 2 or data.shape[1] < window_samples:
                raise ValueError(
                    f"Invalid trial shape {data.shape} for {path}; "
                    f"expected channels x >= {window_samples} samples"
                )
            data = np.asarray(data[:, :window_samples], dtype=np.float64)
            if not np.isfinite(data).all():
                raise ValueError(f"Non-finite samples in: {path}")
            trials.append(data)
            labels.append(label)
            trial_ids.append(f"{label + 1}/{path.name}")
    X = np.stack(trials)
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=4)
    if len(set(counts.tolist())) != 1:
        raise ValueError(f"Class counts must be balanced, got {counts.tolist()}")
    return X, y, trial_ids


def select_usable_channels(X_train, candidates, min_std=1e-6):
    """Return candidate channels with non-degenerate within-trial variance."""
    X_train = np.asarray(X_train)
    selected = []
    for channel in candidates:
        if channel < 0 or channel >= X_train.shape[1]:
            raise ValueError(f"Channel index out of range: {channel}")
        trial_std = np.std(X_train[:, channel, :], axis=-1)
        if np.median(trial_std) > min_std:
            selected.append(int(channel))
    if not selected:
        raise ValueError("No usable channels remain")
    return selected


def chronological_holdout(y, trial_ids, fraction=0.2):
    """Reserve the lexicographically latest trials in every class."""
    y = np.asarray(y)
    if len(y) != len(trial_ids):
        raise ValueError("y and trial_ids must have equal length")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    train_idx, test_idx = [], []
    for label in np.unique(y):
        class_idx = [int(i) for i in np.flatnonzero(y == label)]
        class_idx.sort(key=lambda i: trial_ids[i])
        n_test = max(1, int(round(len(class_idx) * fraction)))
        if n_test >= len(class_idx):
            raise ValueError(f"Class {label} has too few trials for holdout")
        train_idx.extend(class_idx[:-n_test])
        test_idx.extend(class_idx[-n_test:])
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def build_fbtdca(config, window_samples):
    """Construct an FBTDCA estimator from a serializable configuration."""
    passbands = config["passbands"]
    stopbands = config["stopbands"]
    if len(passbands) != len(stopbands):
        raise ValueError("passbands and stopbands must have equal length")
    filterbank = generate_filterbank(passbands, stopbands, SAMPLE_RATE)
    weights = np.asarray(
        [(i + 1) ** (-1.25) + 0.25 for i in range(len(passbands))],
        dtype=float,
    )
    return FBTDCA(
        filterbank=filterbank,
        padding_len=0,
        n_components=int(config["n_components"]),
        filterweights=weights,
    )


def _references(config, window_samples):
    return generate_cca_references(
        TARGET_FREQS,
        srate=SAMPLE_RATE,
        T=window_samples / SAMPLE_RATE,
        n_harmonics=int(config.get("n_harmonics", 5)),
    )


def search_config(X, y, configs, cv_splits=3):
    """Rank FBTDCA configurations using deterministic trial-level CV."""
    X = np.asarray(X)
    y = np.asarray(y)
    if not configs:
        raise ValueError("At least one configuration is required")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []
    for config in configs:
        channels = [int(ch) for ch in config["channels"]]
        Yf = _references(config, X.shape[-1])
        fold_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            estimator = build_fbtdca(config, X.shape[-1])
            estimator.fit(X[train_idx][:, channels].copy(), y[train_idx], Yf=Yf)
            prediction = estimator.predict(X[valid_idx][:, channels].copy())
            fold_scores.append(float(accuracy_score(y[valid_idx], prediction)))
        results.append({
            "config": deepcopy(config),
            "fold_scores": fold_scores,
            "mean_accuracy": float(np.mean(fold_scores)),
            "std_accuracy": float(np.std(fold_scores)),
        })
    results.sort(key=lambda item: (
        -item["mean_accuracy"],
        len(item["config"]["channels"]),
        len(item["config"]["passbands"]),
        int(item["config"]["n_components"]),
        item["config"].get("name", ""),
    ))
    return {
        "best_config": deepcopy(results[0]["config"]),
        "candidate_results": results,
    }


def session_output_dir(session_path, models_root):
    """Return a session-scoped directory, separate from legacy model files."""
    return Path(models_root) / f"browser_{Path(session_path).name}"


def default_configs(usable_channels):
    """Create a compact evidence-based FBTDCA search grid."""
    usable = set(int(ch) for ch in usable_channels)
    channel_sets = {
        "posterior": [ch for ch in [2, 3, 4, 5, 6, 7, 8, 9] if ch in usable],
        "all_valid": sorted(usable),
    }
    banks = {
        "low3": (
            [[6, 45], [14, 45], [22, 45]],
            [[4, 50], [10, 50], [16, 50]],
        ),
        "wide5": (
            [[6, 90], [14, 90], [22, 90], [30, 90], [38, 90]],
            [[4, 100], [10, 100], [16, 100], [24, 100], [32, 100]],
        ),
    }
    configs = []
    for channel_name, channels in channel_sets.items():
        if not channels:
            continue
        for bank_name, (passbands, stopbands) in banks.items():
            for n_components in (2, 4):
                configs.append({
                    "name": f"{channel_name}_{bank_name}_c{n_components}",
                    "channels": channels,
                    "passbands": passbands,
                    "stopbands": stopbands,
                    "n_components": n_components,
                    "n_harmonics": 5,
                })
    return configs


def _sanitize_configs(configs, usable_channels):
    usable = set(usable_channels)
    sanitized = []
    for config in configs:
        candidate = deepcopy(config)
        candidate["channels"] = [
            int(ch) for ch in candidate["channels"] if int(ch) in usable
        ]
        if candidate["channels"]:
            sanitized.append(candidate)
    if not sanitized:
        raise ValueError("No candidate configuration has a usable channel")
    return sanitized


def _fit_and_predict(config, X_train, y_train, X_test):
    window_samples = X_train.shape[-1]
    channels = config["channels"]
    estimator = build_fbtdca(config, window_samples)
    estimator.fit(
        X_train[:, channels].copy(),
        y_train,
        Yf=_references(config, window_samples),
    )
    prediction = estimator.predict(X_test[:, channels].copy())
    return estimator, prediction


def train_session(
    session_path,
    models_root,
    configs=None,
    cv_splits=3,
    window_lengths=(125, 250, 375, 500),
):
    """Tune on training trials, evaluate holdout, and save final FBTDCA models."""
    session_path = Path(session_path)
    window_lengths = tuple(sorted(set(int(v) for v in window_lengths)))
    if not window_lengths or window_lengths[0] <= 0:
        raise ValueError("window_lengths must contain positive integers")
    X, y, trial_ids = load_session(session_path, max(window_lengths))
    train_idx, test_idx = chronological_holdout(y, trial_ids, fraction=0.2)
    usable_channels = select_usable_channels(
        X[train_idx], candidates=range(X.shape[1])
    )
    bad_channels = [ch for ch in range(X.shape[1]) if ch not in usable_channels]
    candidates = default_configs(usable_channels) if configs is None else configs
    candidates = _sanitize_configs(candidates, usable_channels)

    search = search_config(
        X[train_idx, :, :max(window_lengths)],
        y[train_idx],
        candidates,
        cv_splits=cv_splits,
    )
    best_config = search["best_config"]
    output_dir = session_output_dir(session_path, models_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "algorithm": "FBTDCA",
        "session": str(session_path.resolve()),
        "sample_rate_hz": SAMPLE_RATE,
        "target_frequencies_hz": TARGET_FREQS,
        "trial_shape": list(X.shape),
        "class_counts": np.bincount(y, minlength=4).astype(int).tolist(),
        "usable_channels": usable_channels,
        "rejected_channels": bad_channels,
        "median_within_trial_std": np.median(
            np.std(X, axis=-1), axis=0
        ).astype(float).tolist(),
        "split": {
            "method": "last_20_percent_per_class_by_filename",
            "train_trial_ids": [trial_ids[i] for i in train_idx],
            "holdout_trial_ids": [trial_ids[i] for i in test_idx],
        },
        "search": search,
        "selected_config": best_config,
        "windows": {},
    }

    for window_samples in window_lengths:
        X_window = X[:, :, :window_samples]
        evaluation_model, prediction = _fit_and_predict(
            best_config,
            X_window[train_idx],
            y[train_idx],
            X_window[test_idx],
        )
        accuracy = float(accuracy_score(y[test_idx], prediction))
        matrix = confusion_matrix(
            y[test_idx], prediction, labels=np.arange(4)
        ).astype(int)

        final_model, full_prediction = _fit_and_predict(
            best_config,
            X_window,
            y,
            X_window,
        )
        model_path = output_dir / f"model_{window_samples}.pkl"
        joblib.dump(final_model, model_path)
        reloaded = joblib.load(model_path)
        reloaded_prediction = reloaded.predict(
            X_window[:, best_config["channels"]].copy()
        )
        if not np.array_equal(full_prediction, reloaded_prediction):
            raise RuntimeError(f"Reload verification failed for {model_path}")
        metrics["windows"][str(window_samples)] = {
            "holdout_accuracy": accuracy,
            "holdout_confusion_matrix": matrix.tolist(),
            "holdout_predictions": prediction.astype(int).tolist(),
            "holdout_labels": y[test_idx].astype(int).tolist(),
            "model_path": str(model_path.resolve()),
            "reload_verified": True,
        }

    longest = str(max(window_lengths))
    metrics["accuracy_target"] = 0.95
    metrics["accuracy_target_met"] = (
        metrics["windows"][longest]["holdout_accuracy"] >= 0.95
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Browser collection session")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument("--cv-splits", type=int, default=3)
    args = parser.parse_args()
    metrics = train_session(
        args.session,
        args.models_root,
        cv_splits=args.cv_splits,
    )
    print(json.dumps({
        "selected_config": metrics["selected_config"],
        "cross_validation_accuracy": metrics["search"]["candidate_results"][0]["mean_accuracy"],
        "window_holdout_accuracy": {
            key: value["holdout_accuracy"]
            for key, value in metrics["windows"].items()
        },
        "accuracy_target_met": metrics["accuracy_target_met"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
