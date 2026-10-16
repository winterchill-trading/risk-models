# AGENTS.md: lib/

The `risk_models` library. Each module has an `.mli`; change it first and keep it documented.

- `schema.ml`: the ordered feature names and their digest. The digest is the same in Python.
- `encoding.ml`: risk-rules features to a float array, with one-hot columns for the categorical fields.
- `forest.ml`, `linear.ml`, `calibration.ml`: the tree ensemble and the logistic baseline, and the
  isotonic, Platt or identity calibration. The hot functions are `[@zero_alloc]`.
- `lightgbm_dump.ml`: reads LightGBM's `dump_model()` JSON. Follow `../docs/api-notes.md` for the
  missing-value rules.
- `scorer.ml`: loads `model.json` and refuses a schema digest mismatch. `explain` returns risk-rules'
  `Score.t` with the top three reasons.
- `risk_model.ml`: the `Risk_rules.Model.S` adapter; both rules start in shadow mode.
- `tree_shap.ml`: exact per-feature attributions for a `Forest` (path-dependent with per-leaf tables
  and a quadrature fallback, interventional with a background sample). `explanation.ml` is the
  compact, stable record (bin_io and JSON format 1); `explainer.ml` ties them to a `Scorer` or a bare
  `Forest`. Any change must keep `test_tree_shap.ml` (enumerated Shapley values) and
  `test_shap_parity.ml` passing.
- `abstain.ml`: predict-or-abstain by distance from a threshold, accuracy-coverage curves.
- `monitor.ml`: score histogram and PSI, and the Prometheus text. `bench.ml` is the timing harness.
- `json_util.ml`: JSON helpers, kept small.

Rule: a reason-code comparison in a test uses a tie tolerance (see the root AGENTS.md pitfalls).
