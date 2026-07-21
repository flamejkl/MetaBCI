import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from MetaBCIWebGame.train_browser_fbtdca import (
    build_fbtdca,
    chronological_holdout,
    load_session,
    search_config,
    select_usable_channels,
    session_output_dir,
    train_session,
)
from metabci.brainda.algorithms.decomposition.tdca import FBTDCA


class BrowserFBTDCATrainingTest(unittest.TestCase):
    def test_load_session_is_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in range(1, 5):
                class_dir = root / str(label)
                class_dir.mkdir()
                for trial in range(2):
                    data = np.full((3, 20), label * 10 + trial, dtype=float)
                    np.save(class_dir / f"trial_{trial:04d}.npy", data)

            X, y, trial_ids = load_session(root, window_samples=16)

            self.assertEqual(X.shape, (8, 3, 16))
            np.testing.assert_array_equal(np.bincount(y), [2, 2, 2, 2])
            self.assertEqual(trial_ids[0], "1/trial_0000.npy")
            self.assertEqual(trial_ids[-1], "4/trial_0001.npy")

    def test_select_usable_channels_rejects_constant_channel(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(12, 4, 50))
        X[:, 2, :] = -375000.0

        selected = select_usable_channels(X, candidates=[0, 1, 2, 3])

        self.assertEqual(selected, [0, 1, 3])

    def test_chronological_holdout_reserves_last_trial_of_each_class(self):
        y = np.repeat(np.arange(4), 5)
        trial_ids = [f"{label + 1}/trial_{trial:04d}.npy"
                     for label in range(4) for trial in range(5)]

        train_idx, test_idx = chronological_holdout(y, trial_ids, fraction=0.2)

        np.testing.assert_array_equal(np.bincount(y[train_idx]), [4, 4, 4, 4])
        np.testing.assert_array_equal(np.bincount(y[test_idx]), [1, 1, 1, 1])
        self.assertEqual(
            [trial_ids[i] for i in test_idx],
            [f"{label}/trial_0004.npy" for label in range(1, 5)],
        )
        self.assertTrue(set(train_idx).isdisjoint(test_idx))

    def test_build_fbtdca_never_substitutes_another_classifier(self):
        config = {
            "passbands": [[6, 45], [14, 45], [22, 45]],
            "stopbands": [[4, 50], [10, 50], [16, 50]],
            "n_components": 2,
        }

        estimator = build_fbtdca(config, window_samples=500)

        self.assertIsInstance(estimator, FBTDCA)

    def test_search_config_is_deterministic(self):
        rng = np.random.default_rng(11)
        samples = 250
        time = np.arange(samples) / 250.0
        freqs = [8.25, 11.0, 13.75, 16.5]
        trials, labels = [], []
        for label, freq in enumerate(freqs):
            for _ in range(6):
                signal = np.sin(2 * np.pi * freq * time)
                trials.append(np.stack([
                    signal + 0.1 * rng.normal(size=samples),
                    0.8 * signal + 0.1 * rng.normal(size=samples),
                ]))
                labels.append(label)
        X = np.stack(trials)
        y = np.asarray(labels)
        configs = [{
            "name": "one_band",
            "channels": [0, 1],
            "passbands": [[6, 40]],
            "stopbands": [[4, 45]],
            "n_components": 1,
            "n_harmonics": 2,
        }]

        first = search_config(X, y, configs, cv_splits=2)
        second = search_config(X, y, configs, cv_splits=2)

        self.assertEqual(first, second)
        self.assertEqual(first["best_config"]["name"], "one_band")

    def test_session_output_dir_is_scoped_below_models_root(self):
        session = Path("data_self_browser") / "20260720_155846"
        models_root = Path("models")

        result = session_output_dir(session, models_root)

        self.assertEqual(result, models_root / "browser_20260720_155846")

    def test_train_session_saves_reloadable_fbtdca_and_metrics(self):
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = tmp_path / "20260720_155846"
            samples = 150
            time = np.arange(samples) / 250.0
            for label, freq in enumerate([8.25, 11.0, 13.75, 16.5]):
                class_dir = session / str(label + 1)
                class_dir.mkdir(parents=True)
                for trial in range(6):
                    signal = np.sin(2 * np.pi * freq * time)
                    data = np.stack([
                        signal + 0.05 * rng.normal(size=samples),
                        0.7 * signal + 0.05 * rng.normal(size=samples),
                        np.full(samples, -375000.0),
                    ])
                    np.save(class_dir / f"trial_{trial:04d}.npy", data)
            configs = [{
                "name": "test_config",
                "channels": [0, 1, 2],
                "passbands": [[6, 40]],
                "stopbands": [[4, 45]],
                "n_components": 1,
                "n_harmonics": 2,
            }]

            metrics = train_session(
                session,
                tmp_path / "models",
                configs=configs,
                cv_splits=2,
                window_lengths=(125,),
            )

            output_dir = tmp_path / "models" / "browser_20260720_155846"
            self.assertEqual(metrics["algorithm"], "FBTDCA")
            self.assertEqual(metrics["usable_channels"], [0, 1])
            self.assertTrue((output_dir / "metrics.json").is_file())
            estimator = joblib.load(output_dir / "model_125.pkl")
            self.assertIsInstance(estimator, FBTDCA)


if __name__ == "__main__":
    unittest.main()
