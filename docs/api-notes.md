# Notes on the scikit-learn internals the export relies on

Checked with scikit-learn 1.9.1 and numpy 2.5 under Python 3.14 (via uv).

## HistGradientBoostingClassifier

- Trees live in `model._predictors`, a list (one entry per boosting iteration) of
  lists (one predictor per class; one for binary). `predictor.nodes` is a numpy
  structured array with fields `value`, `count`, `feature_idx`, `num_threshold`,
  `missing_go_to_left`, `left`, `right`, `gain`, `depth`, `is_leaf`, `bin_threshold`,
  `is_categorical`, `bitset_idx`. These are private attributes: pin the version when
  retraining.
- Children always have larger indices than their parent, so one reverse sweep fills
  subtree means from `count` and the leaf `value`s.
- Leaf `value` already includes the learning rate. Split-node `value` is not a useful
  expected value (the root is 0); recompute it from counts.
- Prediction: `x <= num_threshold` goes left, NaN follows `missing_go_to_left`. When
  training data has NaN in a feature, a split can separate missing from present values
  with `num_threshold = +inf`; JSON has no infinity, so the export writes the largest
  finite double, which routes every finite value the same way.
- `decision_function` is `_baseline_prediction` plus each tree's leaf in iteration
  order; summing in the same order in OCaml gives bit-identical margins.
- `early_stopping="auto"` turns on for more than 10 000 rows and holds out a random
  validation split; set `early_stopping=False` for a deterministic, time-respecting fit.

## IsotonicRegression

- `X_thresholds_` and `y_thresholds_` are the knots; `predict` clips to
  `[X_min_, X_max_]` (with `out_of_bounds="clip"`) and calls `numpy.interp`.
- `numpy.interp` finds `j` with `xp[j] <= x < xp[j+1]` and returns
  `slope * (x - xp[j]) + fp[j]` with `slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])`.
  Reimplementing exactly that formula agrees to the last bit or within 1e-14.
- Calibrated scores are piecewise constant over large ranges, so "top k per day" and
  threshold choices need ties broken by the raw probability or chosen among the
  values that occur.

## LogisticRegression baseline

- `LogisticRegression` refuses NaN; a `SimpleImputer(strategy="mean")` before the
  `StandardScaler` makes a missing value contribute about zero (the imputer's mean and
  the scaler's mean differ in the last bits), which the OCaml scorer treats as exactly
  zero. Parity stays within 1e-10.
