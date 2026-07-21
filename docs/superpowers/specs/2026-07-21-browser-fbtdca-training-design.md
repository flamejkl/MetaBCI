# Browser SSVEP FBTDCA Training Design

## Goal

Train reproducible four-class SSVEP models from `MetaBCIWebGame/data_self_browser/20260720_155846` with FBTDCA as the only classifier. Target at least 95% accuracy on a trial-level independent chronological holdout when the data supports it; never use train-set scores or duplicated windows as evidence.

## Data and validation

- Load one `.npy` file as one independent trial and preserve the class directory as its label.
- Validate trial shape, finite values, class balance, and channel variance.
- Exclude constant or near-constant channels using statistics computed from training trials only. Channel indices are stored in every artifact.
- Reserve the last 20% of each class by filename order as an untouched chronological holdout. Tune only with stratified cross-validation over the earlier 80%.
- Report cross-validation mean/std, holdout accuracy, confusion matrix, and per-window accuracy.

## Model search

- Use MetaBCI `FBTDCA` exclusively.
- Search a small deterministic grid over valid posterior channel sets, filter banks, harmonic count, and TDCA component count.
- Rank candidates by mean cross-validation accuracy, then prefer the simpler candidate on ties.
- Do not create multiple offset copies of the same trial across train and validation folds.

## Outputs

- Train 125, 250, 375, and 500-sample models with the selected configuration.
- Save artifacts beneath `MetaBCIWebGame/models/browser_20260720_155846/` without replacing current root-level `.pkl` files.
- Save `metrics.json` with dataset identity, split membership, preprocessing, parameters, cross-validation scores, holdout metrics, and model paths.
- Store a wrapper artifact containing the estimator and all inference metadata needed to reproduce preprocessing.

## Failure handling

If the untouched holdout does not reach 95%, report the measured result. Do not tune against the holdout. The acquisition timing and dead/saturated channels are treated as data-quality causes requiring recollection or acquisition fixes rather than hidden by leakage.

## Verification

Unit tests cover deterministic loading/splitting, bad-channel rejection, FBTDCA-only construction, and metadata. The final training run must reload every saved artifact and reproduce its recorded holdout prediction.
