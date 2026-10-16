# AGENTS.md: risk-models

## Purpose

risk-models scores transactions with machine-learned models. Models are trained offline in
Python and exported as small JSON files; the OCaml library loads and scores them and exposes two
rules to risk-rules. The bundled models are trained on synthetic data only, so they show that the
pipeline works and say nothing about real payments.

## Layout

| Path | Holds |
| --- | --- |
| `lib/` | The `risk_models` library: schema, encoding, forest, linear baseline, calibration, LightGBM dump reader, scorer, rule adapter, drift monitor, exact attributions (`Tree_shap`, `Explainer`, `Explanation`), abstention (`Abstain`) |
| `bin/` | The `risk-models` CLI: `schema`, `encode`, `check`, `score`, `explain`, `bench` |
| `models/` | Committed model exports (`model.json`), metrics and `CARD.md` per model |
| `python/` | Training pipeline (`generate.py`, `train.py`, `evaluate.py`, `export.py`), run with `uv`; see `python/AGENTS.md` |
| `test/` | Inline tests, parity fixtures in `test/fixtures/` written by `export.py` |
| `docs/` | `api-notes.md`: scikit-learn and LightGBM internals the export relies on |
| `.github/` | `ci.yml` (OCaml build and test) and `nightly.yml` (full Python pipeline) |

## Build, test, docs

Run from the repository root, inside the workspace (`..` is the workspace root):

```sh
opam exec --switch 5.2.0+ox -- dune build --root .. @risk-models/all
opam exec --switch 5.2.0+ox -- dune build --root .. @risk-models/runtest
```

For parallel work, set `DUNE_BUILD_DIR=_build-risk-models-<task>` (resolved from the workspace
root, because of `--root ..`).

Standalone (outside the workspace): add the organisation opam repository
(`opam repo add winterchill https://github.com/winterchill-trading/opam-repository.git`),
install the dependencies with `opam install --deps-only .`, then `dune build` and `dune test`.
The OCaml tests read the committed fixtures and the Python schema file, so they need no Python.

Python (training, evaluation, export): see `python/AGENTS.md`. Training takes about a minute.

## Conventions

- Format only this repository: `dune build --root .. @risk-models/fmt --auto-promote`. Never run
  `dune fmt` at the workspace root.
- Every module has an `.mli`. Use labelled arguments, `open! Core`, and `ppx_jane`.
- Scoring hot paths are `[@zero_alloc]` (`Forest.margin_u`, `Forest.contributions_u`,
  `Calibration.apply_u`, `Tree_shap.path_dependent_u`, `Tree_shap.interventional_u`). Keep them
  that way and check with `Bench`.
- Attributions are model contributions to the log-odds, not causes. Say so wherever they reach a
  person. `base + contributions = margin` must hold on every fixture row.
- The feature schema belongs to risk-rules. `python/risk_models_train/features.json` is generated
  by `risk-models schema`; a test fails when it and the library disagree. Regenerate, do not hand-edit.
- A retrain is a new `model.json`, reviewed like code. Do not change the rule engine to fit a model.
- Comments for humans: where a test tolerance or a pinned value comes from a platform difference,
  or from a library quirk, say why at that place.
- Do not describe the synthetic models as production-ready in any file.

## Dependencies and dependents

The library depends on `fixed-decimal`, `money`, `ledger`, `payments-core` and `risk-rules` (the
pins in CI). `risk-rules` owns `Features` and the `Model.S` interface; a change there can break the
schema digest here. Check the schema test after any `risk-rules` change.

## CI

`ci.yml` runs the `ox-setup` composite action from `winterchill-trading/infra` with a `pins:` block
(fixed-decimal, money, ledger, payments-core, risk-rules), then `dune build` and `dune test`, plus
the `obj-magic` reusable workflow. `nightly.yml` runs at 03:17 UTC and on demand: `uv sync --locked`,
generate, train, evaluate, export, checks the gradient-boosted model's average precision is above 0.4,
and uploads `out/`. It never replaces the committed models; a person does that.
To check a run, use `gh run list --limit 1`, and at most one `gh run watch <id> --exit-status`.
Do not poll in a loop.

## Pitfalls

- Tree models compare and add only, so they are bit-identical across platforms. Reason codes are not:
  a `log1p` or `exp` on macOS and Linux can differ in the last bit and reorder near-ties. Compare reason
  codes with a tie tolerance, never exactly.
- LightGBM's `dump_model()` prints `internal_value` with 6 digits. Recompute expected values from counts.
  A `missing_type` of `Zero` reads NaN as 0.0 and then `|x| <= 1e-35` follows `default_left`.
- scikit-learn's tree arrays are private attributes. Pin the scikit-learn version when retraining and
  read `docs/api-notes.md` first.
- `export.py` asserts the scorer agrees with scikit-learn; a failing assertion means the export is wrong, not the test.
- shap's interventional mode adds in single precision and goes wrong on rows with a value exactly on a
  split threshold; the shap fixture leaves those rows out of its interventional part and the OCaml
  test allows 1e-6 there (1e-12 for path-dependent). The enumerated-Shapley test covers those rows.
- On OCaml 5 arm64 every store into an `int array` comes with a `dmb ishld` barrier; hot scratch
  integers in `Tree_shap` live in float arrays. Profile before assuming a loop is memory-bound.
- The Python generator re-implements the feature definitions. Training/serving skew is a known risk
  until labelled rows come through `risk-rules features` and `risk-models encode`.

## Where to read more

- `README.md` (quick start, modules, the contract with risk-rules, shadow-first rules, latency and parity).
- `docs/api-notes.md`, `python/AGENTS.md`, `models/*/CARD.md`.
- `../specs/repos/risk-models.md` for the repository summary.
- `../oxcaml-claude-skills/AGENTS.md` (workspace guide) and the skills
  `../oxcaml-claude-skills/skills/oxcaml-performance/` (zero-alloc method) and
  `../oxcaml-claude-skills/skills/banking-standards/` (decisions and audit trails).
- `../oxcaml-claude-skills/GOTCHAS.md`, sections on risk-models.

## Never do

- Install an opam package that changes the compiler, `core`, `base`, `async` or `ppx_jane` on the
  shared `5.2.0+ox` switch. Dry-run first: `opam install --switch 5.2.0+ox --dry-run <pkg>`.
- Replace a committed model or threshold without a retrain, an evaluation and a reviewed diff.
- Publish, tag or release.
- Commit training data that is not synthetic, or any real transaction, label or customer record.
