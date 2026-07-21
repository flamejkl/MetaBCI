# Browser FBTDCA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a leakage-free FBTDCA training pipeline for browser-collected four-class SSVEP data.

**Architecture:** A focused training module owns loading, quality control, deterministic splitting, FBTDCA construction, parameter search, model fitting, and JSON reporting. The command-line entry point trains session-scoped artifacts and verifies them after serialization.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, joblib, MetaBCI FBTDCA, unittest.

## Global Constraints

- FBTDCA is the only classifier.
- One source file is one trial; no derived copies may cross folds.
- Hyperparameters are selected without reading holdout accuracy.
- Existing root-level model files are not overwritten.

---

### Task 1: Dataset loading and quality control

**Files:**
- Create: `MetaBCIWebGame/train_browser_fbtdca.py`
- Create: `tests/test_train_browser_fbtdca.py`

**Interfaces:**
- Produces: `load_session(path, window_samples) -> (X, y, trial_ids)` and `select_usable_channels(X_train, candidates, min_std) -> list[int]`.

- [ ] Write tests using temporary balanced `.npy` trials and a constant channel.
- [ ] Run `python -m unittest tests.test_train_browser_fbtdca -v` and verify missing imports fail.
- [ ] Implement deterministic loading, validation, and bad-channel exclusion.
- [ ] Re-run the tests and verify they pass.

### Task 2: Leakage-free split and FBTDCA construction

**Files:**
- Modify: `MetaBCIWebGame/train_browser_fbtdca.py`
- Modify: `tests/test_train_browser_fbtdca.py`

**Interfaces:**
- Produces: `chronological_holdout(y, trial_ids, fraction) -> (train_idx, test_idx)` and `build_fbtdca(config, window_samples)`.

- [ ] Add tests asserting class balance, no trial overlap, deterministic indices, and an `FBTDCA` estimator.
- [ ] Run the focused tests and verify the new assertions fail.
- [ ] Implement the split, filter-bank/reference construction, and estimator factory.
- [ ] Re-run all focused tests.

### Task 3: Cross-validation search and session-scoped artifacts

**Files:**
- Modify: `MetaBCIWebGame/train_browser_fbtdca.py`
- Modify: `tests/test_train_browser_fbtdca.py`

**Interfaces:**
- Produces: `search_config(X, y, configs, cv) -> dict` and `train_session(session_path, output_root) -> dict`.

- [ ] Add tests for deterministic candidate ranking and non-overwriting output paths.
- [ ] Run the tests and verify failure before implementation.
- [ ] Implement training-fold-only search, four window fits, metrics JSON, and artifact reload checks.
- [ ] Re-run unit tests and the existing test suite.

### Task 4: Train and verify the supplied session

**Files:**
- Create: `MetaBCIWebGame/models/browser_20260720_155846/model_125.pkl`
- Create: `MetaBCIWebGame/models/browser_20260720_155846/model_250.pkl`
- Create: `MetaBCIWebGame/models/browser_20260720_155846/model_375.pkl`
- Create: `MetaBCIWebGame/models/browser_20260720_155846/model_500.pkl`
- Create: `MetaBCIWebGame/models/browser_20260720_155846/metrics.json`

**Interfaces:**
- Consumes: `train_session` from Task 3.
- Produces: loadable model bundles and evidence-backed metrics.

- [ ] Run the CLI against `data_self_browser/20260720_155846`.
- [ ] Inspect cross-validation, chronological holdout, and confusion matrices.
- [ ] Reload all four artifacts and reproduce recorded predictions.
- [ ] Run the complete relevant test suite and record exact results.
