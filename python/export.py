"""Export trained models to the JSON format the OCaml library loads, plus parity fixtures.

Format (format = 1):
  name, kind ("gbdt" | "logistic"), link ("logistic"),
  schema      {name, version, features, digest}
  gbdt        {base_score, trees: [{feature, threshold, missing_left, left, right, value}]}
              feature = -1 marks a leaf; value is the leaf value at leaves and the
              count-weighted expected value of the subtree at split nodes (used for
              reason codes only; the score sums leaf values)
  logistic    {intercept, weights, mean, scale}: margin = intercept + sum w*(x-mean)/scale
  calibration {kind: "isotonic", x, y} | {kind: "platt", a, b} | {kind: "identity"}
  reference   {edges, counts}: calibrated score histogram on the calibration window
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
from train import load, split

# Log-spaced score buckets: almost all traffic scores below 1e-3, and drift that
# matters shows up as mass moving between the low buckets.
SCORE_EDGES = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FIXTURE_ROWS, FIXTURE_POSITIVES, FIXTURE_MISSING = 1000, 150, 20


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
                             mean=sc.mean_.tolist(), scale=sc.scale_.tolist())
    iso = entry["iso"]
    m["calibration"] = dict(kind="isotonic", x=iso.X_thresholds_.tolist(),
                            y=iso.y_thresholds_.tolist())
    counts = np.histogram(ref_scores, SCORE_EDGES)[0]
    m["reference"] = dict(edges=SCORE_EDGES, counts=counts.tolist())
    m["metrics"] = {k: metrics[k] for k in ("pr_auc", "roc_auc", "brier")} | {
        "recall_at_fpr_0.1%": metrics["at_fpr_0.1%"]["recall"],
        "recall_at_fpr_1%": metrics["at_fpr_1%"]["recall"]}
    return m


def sklearn_margin_and_prob(entry, X):
    if "gbdt" in entry:
        g = entry["gbdt"]
        return g.decision_function(X), g.predict_proba(X)[:, 1]
    Z = entry["scaler"].transform(X)
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


def fixture_rows(d, te, missing, rng):
    idx = np.flatnonzero(te)
    pos = idx[d["y"][idx] == 1]
    chosen = np.concatenate([
        rng.choice(pos, FIXTURE_POSITIVES, replace=False),
        rng.choice(idx[d["y"][idx] == 0], FIXTURE_ROWS - FIXTURE_POSITIVES, replace=False)])
    X = d["X"][np.sort(chosen)].copy()
    if missing:
        # A few rows with a missing feature exercise the trees' missing-value branch.
        for r in rng.choice(len(X), FIXTURE_MISSING, replace=False):
            X[r, rng.integers(0, X.shape[1])] = np.nan
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
        is_gbdt = model["kind"] == "gbdt"
        X = fixture_rows(d, te, missing=is_gbdt, rng=rng)
        worst = check_parity(model, entry, X)
        with open(os.path.join(out, "model.json"), "w") as f:
            json.dump(model, f, allow_nan=False, separators=(",", ":"))
            f.write("\n")
        write_fixture(os.path.join(a.fixtures, f"{name}.parity.csv"), model, X)
        size = os.path.getsize(os.path.join(out, "model.json"))
        print(f"{name}: {size // 1024} KB, reference vs scikit-learn max diff {worst:.2e}")


if __name__ == "__main__":
    main()
