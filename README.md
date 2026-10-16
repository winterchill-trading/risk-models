# risk-models

Machine-learned scores for transaction monitoring and fraud. Models are trained
offline in Python and scored in OCaml from a small committed JSON export. The
features are the ones risk-rules defines (`Risk_rules.Features`), and the scores
reach the decision through risk-rules' `Model.S` interface as two rules: Review and
Block, started in shadow mode.

> **Synthetic data only.** The bundled models were trained and evaluated on data the
> generator in `python/generate.py` makes up, with fraud patterns it injects itself.
> The metrics below show that the pipeline works end to end and how well the features
> separate those made-up patterns. They say nothing about performance on real
> payments. Retrain, re-evaluate and re-choose the thresholds on real labelled
> traffic before relying on a model.

## Why separate

Rules and models change on different clocks and are owned by different people.
risk-rules holds the deterministic rules and the single definition of every feature;
this repo holds what is learned from data: the training pipeline, the exported
parameters, the scorer and the drift monitor. Keeping them apart means a retrain is a
new JSON file (reviewed like code, versioned by its digest) and never a code change in
the rule engine, while the feature contract between the two is checked by a schema
digest at load time and in CI.

## Quick start

Train (Python through `uv`, about a minute on a laptop):

```sh
cd python
uv sync
uv run python generate.py     # data/synthetic.npz, 6000 customers, 91 days
uv run python train.py        # gradient-boosted trees and a logistic baseline
uv run python evaluate.py     # ../models/<name>/metrics.json and CARD.md
uv run python export.py       # ../models/<name>/model.json, ../test/fixtures/*.csv
```

Score in OCaml:

```ocaml
open! Core
open Risk_models

let model = Scorer.load "models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn

(* A Risk_rules.Features.t, computed by risk-rules from a Context. *)
let decide features =
  let x = Encoding.encode features in
  let p = Scorer.score model x in                    (* calibrated probability *)
  let s = Scorer.explain model x in    (* Risk_rules.Model.Score.t, top 3 reasons *)
  p, s.reasons

(* Exact per-feature contributions for a reviewer: base + contributions = margin. *)
let explainer = Explainer.create model |> Or_error.ok_exn   (* one per thread *)
let why features =
  Explainer.explain ~k:5 explainer (Encoding.encode features)
  |> Explanation.to_json_string        (* stable JSON, format 1 *)

(* Abstain near the threshold: predict only when the score is far enough from it. *)
let abstain = Abstain.create ~threshold:0.01 ~confidence:Logit_margin ~min_confidence:0.5
let decide_or_defer score = Abstain.decide (Or_error.ok_exn abstain) score

(* Or hand the model to risk-rules: two rules, capped at Allow until promoted. *)
let rules = Risk_model.rules ~shadow:true model     (* model_review, model_block *)
let specs = Risk_model.rule_specs model             (* the same as policy sexp *)
```

CLI:

```sh
risk-models schema                     # the feature schema as JSON (python/ reads it)
risk-rules features ctx.sexp | risk-models encode > rows.csv   # contexts to model rows
risk-models check MODEL.json           # load, check the schema digest, summarise
risk-models score MODEL.json rows.csv  # score and reasons per CSV row
risk-models explain -model MODEL.json -features amount=1200,beneficiary_age_hours=2
                                       # explanation JSON: base, margin, top k, rest
risk-models explain -model MODEL.json -csv rows.csv -background bg.csv -k 3
                                       # interventional, against background rows
risk-models bench MODEL.json rows.csv  # latency and words per call
```

No environment variables, credentials or network access are needed.

## Modules

| Module | What it is |
|--------|------------|
| `Schema` | ordered feature names with a digest (hex MD5 of the names joined by newlines, identical in Python) |
| `Encoding` | `Risk_rules.Features.t -> float array`: `Features.to_floats` order, then one-hot columns for `rail`, `card_channel`, `payee_check` |
| `Forest` | tree ensemble in flat arrays with per-split missing-value modes; `margin_u` and `contributions_u` are `[@zero_alloc]` |
| `Linear` | the logistic baseline (log1p on heavy-tailed features, missing features contribute nothing) |
| `Calibration` | isotonic (numpy `interp` semantics), Platt or identity; `apply_u` is `[@zero_alloc]` |
| `Lightgbm_dump` | reads LightGBM `Booster.dump_model()` JSON into a `Forest` (binary objective, numerical splits, `missing_type` NaN/None/Zero) |
| `Scorer` | loads `model.json`, refuses a schema digest mismatch; margin, probability, calibrated score, `explain` returning a `Risk_rules.Model.Score.t` |
| `Tree_shap` | exact additive per-feature attributions for a `Forest`: path-dependent (from the node covers, tabulated per leaf, `[@zero_alloc]`) and interventional (against a background sample, `[@zero_alloc]`) |
| `Explanation` | a compact explanation: base, margin, score, top k contributions with reviewer-facing labels, the rest summed; `Stable.V1` bin_io and a stable JSON form |
| `Explainer` | attributions and explanations for a `Scorer` (trees, or the logistic model's weighted terms) or for any bare `Forest` with feature names |
| `Abstain` | selective classification: predict or abstain by distance from the threshold, accuracy-coverage curves from labelled scores, the cut-off for a target coverage |
| `Risk_model` | `Risk_rules.Model.S` adapter, `model_review` and `model_block` rules, shadow mode, policy specs |
| `Monitor` | live score histogram, PSI against the calibration window, Stable/Review/Retrain, Prometheus text |
| `Bench` | scoring latency and allocation |

## The contract with risk-rules

- risk-rules `Features` defines every feature once. `Encoding.schema` is
  `Features.numeric_names` plus the one-hot columns; `risk-models schema` writes it to
  `python/risk_models_train/features.json`, which the Python side reads. A test fails
  when the committed file and the library disagree, so a risk-rules schema change
  shows up in CI here.
- Every `model.json` records the schema digest; `Scorer.load` refuses a model trained
  on another layout.
- `Risk_model.to_risk_rules` is a `Risk_rules.Model.S`: name, version (first 12 hex
  digits of the MD5 of the model JSON), and `score : Features.t -> Score.t` with the
  top three reasons.
- The generator mirrors the meaning of each risk-rules field (windows, units, NaN for
  missing) in Python. That is a second implementation of the features and a known
  source of training/serving skew. For real data the path is already there:
  `risk-rules features` turns a journaled `Context` into `Features` JSON through
  `Features.of_context`, and `risk-models encode` turns those lines into CSV rows in
  model order (a test scores the risk-rules example contexts this way). Joining
  labels and timestamps to those rows for `train.py` is the missing step.

## Two rules, shadow first

The model's calibrated score becomes two rules:

- `model_review` (Review) at the lowest score that kept the calibration window within
  the daily alert budget (50 a day);
- `model_block` (Block) at the lowest score whose precision on the calibration window
  reached 0.9 (absent when no score did).

Both thresholds ship in `model.json` and are chosen by `evaluate.py`. `Risk_model.rules`
caps both at Allow by default (`Risk_rules.Rule.cap`), and `rule_specs` wraps them in
`Shadow`: hits are journaled with score, model name, version and reasons but change no
decision. Promote after two weeks of shadow running with score PSI below 0.1 and alert
counts near the budget.

## Model format

`model.json` (format 1), written by `python/export.py`:

- `schema`: name, version, feature names, digest
- `kind`: `gbdt` (scikit-learn trees as parallel arrays: `feature` (-1 = leaf),
  `threshold`, `missing_left`, `left`, `right`, `value`), `lightgbm` (the booster's
  `dump_model()` with gains and weights stripped) or `logistic` (`intercept`,
  `weights`, `mean`, `scale`, `log1p`)
- `calibration`: isotonic knots
- `reference`: score histogram on the calibration window, for `Monitor`
- in `gbdt` trees, `count`: training rows per node (optional; path-dependent attributions
  need it, LightGBM dumps carry their own counts)
- `thresholds`: review and block, with the budget and precision target they were chosen for

Tree semantics follow scikit-learn: `x <= threshold` goes left, NaN follows
`missing_left`, the margin is `base_score` plus the leaf values in tree order. Split
node `value`s are count-weighted subtree means, used only for reason codes (path
attribution: each split credits the change in expected value to its feature, so bias
plus contributions equals the margin exactly). A threshold of +inf (missing versus
present) is written as the largest finite double.

## Explanations

`Tree_shap` computes the exact Shapley values of the tree ensemble's margin: one
contribution per feature, adding up with the base value to the margin
(`base + sum = margin`, checked on every fixture row). Two definitions are offered, and
they differ when features are correlated:

- path-dependent (default): features outside a subset follow the training data down each
  tree in proportion to the node covers, so the base is the cover-weighted mean margin.
  This is the exact tree-path algorithm (TreeSHAP-style). Its per-leaf results depend on
  the row only through which of the leaf's path features the row follows, so the
  workspace tabulates every leaf with at most 10 distinct path features for all such
  patterns (4.2 MB for the 150-tree model) and computes deeper leaves at run time by an
  exact Gauss-Legendre quadrature of the Shapley integral. Trees may have up to 63 leaves.
- interventional: features outside a subset take their values from each row of a
  background sample; the base is the background's mean margin. Cost grows with the sample.

`Explanation.t` keeps the top k (default 5) by absolute contribution, with
reviewer-facing labels, plus the sum of the rest. These are model contributions to the
log-odds, not causes. The JSON form (format 1) has a fixed field order and `null` for a
missing feature value:

```json
{"format":1,"model":"txn-fraud-gbdt","version":"66c8ce496731","method":"path_dependent",
 "base":-9.6926,"margin":-3.3673,"score":0.0935,
 "top":[{"feature":"beneficiary_age_hours","label":"beneficiary_age_hours","value":2.0,
         "contribution":3.0265}, ...],
 "rest":-0.7079}
```

The existing reason codes (`Scorer.explain`, `Risk_model`) are unchanged: they are path
attributions, cheaper and stable, and the parity tests pin them.

Parity: `python/export.py` writes `test/fixtures/<model>.shap.csv` from the shap library's
`TreeExplainer` (0.53) on the exported JSON trees (scikit-learn) or the booster
(LightGBM). The OCaml values match:

| | path-dependent (200 rows) | interventional (50 rows, 100 background) |
|---|---|---|
| scikit-learn trees | 3.6e-15 | 8.5e-8 |
| LightGBM | 1.8e-15 | 4.6e-7 |

shap's interventional code adds in single precision (a leaf of 1 + 1e-12 comes back as
exactly 1), which is the whole difference there; the test bounds are 1e-12 and 1e-6. Its
interventional values also stop adding up for rows with a value exactly on a split
threshold, so the export leaves those rows out of that part. Independently of shap, a
test enumerates every feature subset of 200 random small forests (with missing values,
repeated features and values on thresholds) and checks both variants against the
Shapley formula to 1e-12.

## Abstaining

`Abstain` wraps a thresholded decision: with confidence the distance of the score from
the threshold (in log-odds or in probability), scores closer than `min_confidence`
abstain. `Abstain.Curve.compute` gives, from labelled scores, accuracy, precision and
recall at every coverage; `for_coverage` picks `min_confidence` for a target coverage,
and `band` gives the abstaining scores as an interval. In a decision system the abstain
band is a review queue or a fallback to rules; transaction-monitoring uses it both ways.

## Results on the synthetic set

Time-ordered split with a 7-day label delay gap: days 0-51 train, 52-65 calibrate,
66-72 dropped, 73-90 test (146 504 rows, 504 positive, 0.34%).

| | scikit-learn trees | LightGBM | Logistic baseline |
|---|---|---|---|
| Average precision | 0.547 | 0.544 | 0.436 |
| ROC-AUC | 0.979 | 0.980 | 0.972 |
| Precision at 25 / 50 / 100 alerts a day | 0.618 / 0.422 / 0.282 | 0.620 / 0.416 / 0.280 | 0.564 / 0.381 / 0.267 |
| Recall at 50 alerts a day | 0.647 | 0.635 | 0.565 |
| Recall at 0.1% / 1% FPR | 0.498 / 0.736 | 0.486 / 0.726 | 0.369 / 0.635 |
| Review rule on test (alerts a day, precision, recall) | 36.8, 0.42, 0.61 | 36.0, 0.43, 0.61 | 31.3, 0.42, 0.53 |
| Block rule on test (alerts a day, precision, recall) | 10.8, 0.81, 0.35 | 10.6, 0.84, 0.35 | 4.9, 0.81, 0.16 |
| Prequential average precision (4 weekly folds) | 0.601 +/- 0.036 | | |

Recall at 1% FPR per typology (gradient-boosted): card-not-present bursts 0.51,
account takeover 1.00, money mules 0.95, structuring 1.00, push payment fraud 0.92.
Card bursts are weakest because risk-rules' outgoing counts and sums cover transfers
only; card velocity comes from `card_auths_10m` and `card_declines_10m`. The model
cards (`models/*/CARD.md`) have the full tables, cost figures and drift checks.

## Latency and parity

On an Apple M-series laptop, release build, rows from the parity fixture:

| | `Scorer.score_u` | `Scorer.explain` | attributions (`Explainer.contributions_u`) | `Explainer.explain` (top 5) | interventional, 100 background rows |
|---|---|---|---|---|---|
| scikit-learn trees (150 trees, 4350 nodes) | 2.1 us, 0 words | 3.4 us, 480 words | 32 us, 0 words | 34 us, 666 words | 0.96 ms, 0 words |
| LightGBM (150 trees, 4350 nodes) | 2.5 us, 0 words | 3.8 us, 520 words | 32 us, 0 words | 34 us, 668 words | 0.93 ms, 0 words |
| Logistic | 0.18 us, 0 words | 0.67 us, 574 words | 0.13 us, 0 words | 2.0 us, 687 words | |

The attribution numbers were measured with the machine under load from other builds
(the score column read 2.8 us in the same run). Path-dependent attributions started at
170 us with the textbook node-by-node walk; tabulating leaves took them to about 30 us,
which is now the cost of adding about 9 400 table entries per row (four per leaf), short
of the few microseconds the scores take. They are meant for decisions a person reviews,
not for every payment. The transaction-monitoring model (102 trees, leaves with up to 14
distinct features) takes about 65 us with a 27 MB table, or 90 us with
`~max_table_depth:8` and 8.5 MB.

Per-call percentiles from `risk-models bench` are quantised to 1 us by the macOS clock.
Scoring and feeding the score into `Monitor.observe_u` allocates nothing (an
expect test pins 0 words).

Parity: `test/test_parity.ml` scores the 1000 committed rows per model (with missing
values and values exactly at split thresholds) and requires margin, probability and
calibrated score within 1e-9 of the Python reference (the research note asks for 1e-6)
and the same reason codes (two codes may swap only when their contributions are
within 1e-12: Linux and macOS `log1p` differ in the last bit). The scikit-learn and logistic models match bit for bit on
macOS arm64, LightGBM within 1.1e-16; the
tolerance leaves room for a last-bit `exp` difference between platform maths
libraries. The Python reference itself matches scikit-learn within 1e-10 before
export.

## CI

`ci.yml` builds and tests the OCaml side on the committed models and fixtures, pinning
fixed-decimal, money, ledger, payments-core and risk-rules. `nightly.yml` runs the
whole Python pipeline (about a minute) and checks an average precision floor; it never
replaces the committed models.

## Known gaps

- Synthetic data only. No public transaction-monitoring dataset is downloadable
  without registration (PaySim and the card-fraud sets need a Kaggle login), so there
  is no loader for one.
- The generator re-implements the risk-rules feature definitions in Python instead of
  exporting rows through `Features.of_context`.
- scikit-learn HistGradientBoosting and LightGBM are read; an XGBoost JSON parser
  (strict less-than splits compared in float32) is not written yet. The LightGBM
  parity fixture has extra missing values but no rows placed exactly on split
  thresholds (the scikit-learn one has 40).
- Reason codes (`Scorer.explain`, the risk-rules rules) are still path attributions;
  exact attributions are a separate call (`Explainer`) at about ten times the cost of a
  score.
- Path-dependent attributions need trees of at most 63 leaves and the node counts in the
  export (older exports without them only explain interventionally).
