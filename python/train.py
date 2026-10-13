"""Train the gradient-boosted model and the logistic baseline, then calibrate both.

Splits are by time, as in production (risk_models_train/splits.py): days 0-51 train,
days 52-65 calibrate (isotonic regression on the model's probability), days 66-72 are a
7-day label delay gap and are never used, days 73 onwards are held out for evaluate.py.
Training is deterministic (no early stopping, fixed seeds).

Usage: uv run python train.py [--data data/synthetic.npz] [--out data/trained.pkl]
"""

import argparse
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from risk_models_train import schema
from risk_models_train.splits import holdout as split


def make_gbdt():
    # Small, shallow trees keep the exported model a few hundred KB and scoring a few
    # microseconds; deeper trees did not improve average precision on the synthetic set.
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def load(path):
    d = np.load(path)
    if str(d["digest"]) != schema.digest():
        raise SystemExit(f"{path} was generated with another feature schema; regenerate it")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/synthetic.npz")
    p.add_argument("--out", default="data/trained.pkl")
    a = p.parse_args()
    d = load(a.data)
    X, y = d["X"], d["y"]
    tr, ca, _ = split(d["t"])
    print(f"train {tr.sum()} rows ({y[tr].sum()} positive), calibrate {ca.sum()} rows")

    start = time.time()
    gbdt = make_gbdt()
    gbdt.fit(X[tr], y[tr])
    print(f"gbdt: {gbdt.n_iter_} trees in {time.time() - start:.1f} s")

    start = time.time()
    scaler = StandardScaler().fit(X[tr])
    logit = LogisticRegression(C=1.0, max_iter=2000)
    logit.fit(scaler.transform(X[tr]), y[tr])
    print(f"logistic baseline in {time.time() - start:.1f} s")

    models = {}
    for name, raw in [
        ("txn-fraud-gbdt", lambda Z: gbdt.predict_proba(Z)[:, 1]),
        ("txn-fraud-logistic", lambda Z: logit.predict_proba(scaler.transform(Z))[:, 1]),
    ]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw(X[ca]), y[ca])
        models[name] = dict(iso=iso)
    models["txn-fraud-gbdt"]["gbdt"] = gbdt
    models["txn-fraud-logistic"].update(scaler=scaler, logit=logit)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(models=models, data=a.data, digest=schema.digest()), f)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
