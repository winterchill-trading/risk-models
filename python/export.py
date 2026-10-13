"""Export trained models to the JSON format the OCaml library loads, plus parity fixtures.

Format (format = 1):
  name, kind ("gbdt" | "logistic"), link ("logistic"),
  schema      {name, version, features, digest}
  gbdt        {base_score, trees: [{feature, threshold, missing_left, left, right, value}]}
              feature = -1 marks a leaf; value is the leaf value at leaves and the
              count-weighted expected value of the subtree at split nodes (used for
              reason codes only; the score sums leaf values)
  logistic    {intercept, weights, mean, scale, log1p}: x' = log1p(max(x, 0)) where log1p
              is 1, then margin = intercept + sum w*(x'-mean)/scale over present
              features; a missing (NaN) feature contributes nothing
  calibration {kind: "isotonic", x, y} | {kind: "platt", a, b} | {kind: "identity"}
  reference   {edges, counts}: calibrated score histogram on the calibration window
  thresholds  {review, block, alerts_per_day, block_precision}: calibrated-score cut-offs
              for the model_review and model_block rules (block is null when no score
              reached the precision target), chosen by evaluate.py
  metrics     a summary copied from metrics.json, informational

Before writing, the reference scorer (risk_models_train/scorer.py) must agree with
scikit-learn's own predictions on the test window; the parity fixture then records
1000 rows with the reference scorer's outputs for the OCaml tests.

Usage: uv run python export.py [--data data/synthetic.npz] [--trained data/trained.pkl]
"""

import argparse
import csv
import json
import os
import pickle

import numpy as np

from risk_models_train import schema, scorer
from evaluate import linear_view
from train import load, split

# Log-spaced score buckets: almost all traffic scores below 1e-3, and drift that
# matters shows up as mass moving between the low buckets.
SCORE_EDGES = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FIXTURE_ROWS, FIXTURE_POSITIVES, FIXTURE_MISSING, FIXTURE_AT_SPLIT = 1000, 150, 20, 40


def export_tree(pred):
    nodes = pred.nodes
    n = len(nodes)
    leaf = nodes["is_leaf"].astype(bool)
    feature = np.where(leaf, -1, nodes["feature_idx"]).astype(int)
    left = np.where(leaf, -1, nodes["left"].astype(np.int64))
    right = np.where(leaf, -1, nodes["right"].astype(np.int64))
    value = nodes["value"].astype(float).copy()
    count = nodes["count"].astype(float)
    # Children always have larger indices than their parent, so a reverse sweep fills
    # every split node's expected value from its children.
    for i in range(n - 1, -1, -1):
        if not leaf[i]:
            l, r = left[i], right[i]
            value[i] = (count[l] * value[l] + count[r] * value[r]) / (count[l] + count[r])
    threshold = np.where(leaf, 0.0, nodes["num_threshold"])
    # A split that separates missing values from all present ones has threshold +inf
    # (every present value goes left). JSON has no infinity; the largest finite double
    # routes every finite input the same way.
    threshold = np.where(threshold == np.inf, np.finfo(np.float64).max, threshold)
    assert np.isfinite(threshold).all(), "non-finite split threshold"
    assert not nodes["is_categorical"].any()
    return dict(
        feature=feature.tolist(), threshold=threshold.tolist(),
        missing_left=nodes["missing_go_to_left"].astype(int).tolist(),
        left=left.tolist(), right=right.tolist(), value=value.tolist())


def to_json(name, entry, metrics, ref_scores):
    m = dict(
        format=1, name=name, link="logistic",
        schema=dict(name=schema.SCHEMA_NAME, version=schema.SCHEMA_VERSION,
                    features=schema.FEATURES, digest=schema.digest()))
    if "gbdt" in entry:
        g = entry["gbdt"]
        m["kind"] = "gbdt"
        m["gbdt"] = dict(base_score=float(g._baseline_prediction.ravel()[0]),
                         trees=[export_tree(p) for (p,) in g._predictors])
    else:
        sc, lr = entry["scaler"], entry["logit"]
        m["kind"] = "logistic"
        m["logistic"] = dict(intercept=float(lr.intercept_[0]), weights=lr.coef_[0].tolist(),
                             mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                             log1p=[int(f in schema.LOG_FEATURES) for f in schema.FEATURES])
    iso = entry["iso"]
    m["calibration"] = dict(kind="isotonic", x=iso.X_thresholds_.tolist(),
                            y=iso.y_thresholds_.tolist())
    counts = np.histogram(ref_scores, SCORE_EDGES)[0]
    m["reference"] = dict(edges=SCORE_EDGES, counts=counts.tolist())
    r = metrics["rules"]
    m["thresholds"] = dict(review=r["review"], block=r["block"],
                           alerts_per_day=r["review_budget_per_day"],
                           block_precision=r["block_precision"])
    m["metrics"] = {k: metrics[k] for k in ("pr_auc", "roc_auc", "brier")} | {
        "recall_at_fpr_0.1%": metrics["at_fpr_0.1%"]["recall"],
        "recall_at_fpr_1%": metrics["at_fpr_1%"]["recall"]}
    return m


def sklearn_margin_and_prob(entry, X):
    if "gbdt" in entry:
        g = entry["gbdt"]
        return g.decision_function(X), g.predict_proba(X)[:, 1]
    Z = linear_view(entry, X)
    return entry["logit"].decision_function(Z), entry["logit"].predict_proba(Z)[:, 1]


def check_parity(model, entry, X):
    margin, prob = sklearn_margin_and_prob(entry, X)
    cal = entry["iso"].predict(prob)
    worst = 0.0
    for i in range(len(X)):
        mg, p, s, _ = scorer.score(model, X[i].tolist())
        worst = max(worst, abs(mg - margin[i]), abs(p - prob[i]), abs(s - cal[i]))
    assert worst < 1e-10, f"reference scorer disagrees with scikit-learn by {worst}"
    return worst


def fixture_rows(d, te, model, rng):
    idx = np.flatnonzero(te)
    pos = idx[d["y"][idx] == 1]
    chosen = np.concatenate([
        rng.choice(pos, FIXTURE_POSITIVES, replace=False),
        rng.choice(idx[d["y"][idx] == 0], FIXTURE_ROWS - FIXTURE_POSITIVES, replace=False)])
    X = d["X"][np.sort(chosen)].copy()
    rows = rng.choice(len(X), FIXTURE_MISSING + FIXTURE_AT_SPLIT, replace=False)
    # Extra missing values on top of the data's own, so every model sees the branch.
    for r in rows[:FIXTURE_MISSING]:
        X[r, rng.integers(0, X.shape[1])] = np.nan
    if model["kind"] == "gbdt":
        # Values exactly equal to a split threshold, where "<=" versus "<" would send
        # the row the other way.
        trees = model["gbdt"]["trees"]
        for r in rows[FIXTURE_MISSING:]:
            tree = trees[rng.integers(0, len(trees))]
            splits = [i for i, f in enumerate(tree["feature"])
                      if f >= 0 and tree["threshold"][i] < 1e300]
            i = splits[rng.integers(0, len(splits))]
            X[r, tree["feature"][i]] = tree["threshold"][i]
    return X


def write_fixture(path, model, X):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(schema.FEATURES + ["margin", "probability", "score", "reasons"])
        for x in X:
            mg, p, s, rs = scorer.score(model, x.tolist())
            w.writerow([repr(float(v)) for v in x] + [repr(mg), repr(p), repr(s),
                                                      ";".join(map(str, rs))])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/synthetic.npz")
    p.add_argument("--trained", default="data/trained.pkl")
    p.add_argument("--models", default="../models")
    p.add_argument("--fixtures", default="../test/fixtures")
    a = p.parse_args()
    d = load(a.data)
    # The pickle is the local output of train.py, never a downloaded file.
    with open(a.trained, "rb") as f:
        trained = pickle.load(f)
    _, ca, te = split(d["t"])
    for name, entry in trained["models"].items():
        rng = np.random.default_rng(11)
        out = os.path.join(a.models, name)
        with open(os.path.join(out, "metrics.json")) as f:
            metrics = json.load(f)
        _, prob = sklearn_margin_and_prob(entry, d["X"][ca])
        model = to_json(name, entry, metrics, entry["iso"].predict(prob))
        X = fixture_rows(d, te, model, rng)
        worst = check_parity(model, entry, X)
        with open(os.path.join(out, "model.json"), "w") as f:
            json.dump(model, f, allow_nan=False, separators=(",", ":"))
            f.write("\n")
        write_fixture(os.path.join(a.fixtures, f"{name}.parity.csv"), model, X)
        size = os.path.getsize(os.path.join(out, "model.json"))
        print(f"{name}: {size // 1024} KB, reference vs scikit-learn max diff {worst:.2e}")


if __name__ == "__main__":
    main()
