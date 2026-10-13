"""Evaluate trained models on the held-out window and write each model card.

Metrics (on calibrated scores, test window = last 20% of time):
  PR-AUC (average precision) and ROC-AUC
  recall at 0.1% and 1% false-positive rate, overall and per typology
  Brier score and expected calibration error (10 equal-count bins)
  cost: a threshold chosen on the calibration window to minimise
        missed loss + review cost, then applied to the test window
  drift: population stability index of every feature and of the score between the
        training and test windows (PSI > 0.2 is flagged)
  importance: split gain per feature (trees) or |weight| (logistic)

Writes ../models/<name>/metrics.json and ../models/<name>/CARD.md.

Usage: uv run python evaluate.py [--data data/synthetic.npz] [--trained data/trained.pkl]
"""

import argparse
import datetime
import json
import os
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from risk_models_train import schema
from train import load, split

REVIEW_COST = 4.0  # EUR per alert an analyst looks at
# Money-laundering typologies have no direct loss to the bank; a missed case is priced
# as a flat regulatory exposure instead of the amount.
AML_MISS_COST = 500.0
AML = {schema.TYPOLOGIES.index("money_mule"), schema.TYPOLOGIES.index("structuring")}


def scores(entry, X):
    if "gbdt" in entry:
        raw = entry["gbdt"].predict_proba(X)[:, 1]
    else:
        raw = entry["logit"].predict_proba(entry["scaler"].transform(X))[:, 1]
    return entry["iso"].predict(raw)


def threshold_at_fpr(s, y, fpr):
    neg = np.sort(s[y == 0])
    return neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]


def recall_at(s, y, typ, fpr):
    th = threshold_at_fpr(s, y, fpr)
    out = {"threshold": float(th), "recall": float((s[y == 1] > th).mean())}
    for k, name in enumerate(schema.TYPOLOGIES[1:], start=1):
        m = typ == k
        if m.any():
            out[name] = float((s[m] > th).mean())
    return out


def miss_cost(typ, amount):
    return np.where(np.isin(typ, list(AML)), AML_MISS_COST, amount) * (typ > 0)


def cost(s, typ, amount, th):
    alert = s > th
    return float(miss_cost(typ, amount)[~alert].sum() + REVIEW_COST * alert.sum())


def best_threshold(s, typ, amount):
    grid = np.unique(np.quantile(s, np.linspace(0.9, 0.99999, 400)))
    costs = [cost(s, typ, amount, th) for th in grid]
    return float(grid[int(np.argmin(costs))])


def ece(s, y, bins=10):
    order = np.argsort(s)
    total = 0.0
    for chunk in np.array_split(order, bins):
        total += len(chunk) * abs(s[chunk].mean() - y[chunk].mean())
    return float(total / len(s))


def psi(ref, cur, bins=10):
    # Quantile edges from the reference window; a feature with a large point mass (a
    # flag, or zero for card payments) collapses to fewer bins, which is what we want.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]]) if len(edges) > 2 \
        else np.array([-np.inf, edges[0], np.inf])
    # Right-closed bins so a point mass at an edge lands in one bin.
    r = np.bincount(np.searchsorted(edges, ref, side="left"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="left"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


def importance(entry):
    if "gbdt" in entry:
        gain = np.zeros(len(schema.FEATURES))
        for (pred,) in entry["gbdt"]._predictors:
            nodes = pred.nodes
            split_nodes = nodes[nodes["is_leaf"] == 0]
            np.add.at(gain, split_nodes["feature_idx"], split_nodes["gain"])
        return gain / gain.sum()
    w = np.abs(entry["logit"].coef_[0])
    return w / w.sum()


def evaluate(name, entry, d):
    X, y, typ, amount = d["X"], d["y"], d["typology"], d["amount"]
    tr, ca, te = split(d["t"])
    s_te, s_ca, s_tr = scores(entry, X[te]), scores(entry, X[ca]), scores(entry, X[tr])
    th = best_threshold(s_ca, typ[ca], amount[ca])
    m = {
        "name": name,
        "rows": {"train": int(tr.sum()), "calibration": int(ca.sum()), "test": int(te.sum())},
        "positives_test": int(y[te].sum()),
        "pr_auc": float(average_precision_score(y[te], s_te)),
        "roc_auc": float(roc_auc_score(y[te], s_te)),
        "brier": float(brier_score_loss(y[te], s_te)),
        "ece": ece(s_te, y[te]),
        "at_fpr_0.1%": recall_at(s_te, y[te], typ[te], 0.001),
        "at_fpr_1%": recall_at(s_te, y[te], typ[te], 0.01),
        "cost": {
            "review_cost_eur": REVIEW_COST,
            "aml_miss_cost_eur": AML_MISS_COST,
            "threshold": th,
            "alerts": int((s_te > th).sum()),
            "model_eur": cost(s_te, typ[te], amount[te], th),
            "no_model_eur": float(miss_cost(typ[te], amount[te]).sum()),
        },
        "drift_psi": {
            "score": psi(s_tr, s_te),
            **{f: psi(X[tr][:, i], X[te][:, i]) for i, f in enumerate(schema.FEATURES)},
        },
        "importance": dict(zip(schema.FEATURES, map(float, importance(entry)))),
    }
    return m


def card(m, entry):
    kind = "gradient-boosted trees (scikit-learn HistGradientBoosting)" if "gbdt" in entry \
        else "logistic regression on standardised features"
    f1, f01 = m["at_fpr_1%"], m["at_fpr_0.1%"]
    top = sorted(m["importance"].items(), key=lambda kv: -kv[1])[:8]
    drift = {k: v for k, v in m["drift_psi"].items() if v > 0.2}
    c = m["cost"]
    lines = [
        f"# {m['name']}",
        "",
        f"Generated by `python/evaluate.py` on {datetime.date.today().isoformat()}.",
        "",
        "## What it is",
        "",
        f"A {kind} scoring one outgoing payment (card authorisation or transfer) for",
        "fraud and money-laundering typologies. The output is a calibrated probability",
        "(isotonic regression fitted on a later time window than training) with up to",
        "three reason codes: the features that pushed the score up the most.",
        "",
        "## Intended use",
        "",
        "A second opinion next to the deterministic rules in risk-rules: route to review,",
        "add friction (step-up authentication, a confirmation-of-payee warning), or rank",
        "an alert queue. It should not block a payment on its own, and it has only seen",
        "synthetic data: thresholds must be re-chosen on real traffic before use.",
        "",
        "## Data",
        "",
        "Synthetic only (`python/generate.py`). No public transaction-monitoring dataset",
        "is downloadable without registration (PaySim and the card-fraud sets need a",
        "Kaggle login), so the generator models customers with daily and weekly seasonality",
        "and injects five typologies: card-not-present bursts, account takeover, mule",
        "chains, structuring under the 10 000 EUR threshold and authorised push payment",
        "scams. Splits are by time: train 65%, calibration 15%, test 20%.",
        "",
        f"Rows: {m['rows']['train']} train, {m['rows']['calibration']} calibration,",
        f"{m['rows']['test']} test ({m['positives_test']} positive in test).",
        "",
        "## Metrics (test window)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| PR-AUC | {m['pr_auc']:.4f} |",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| Recall at 0.1% FPR | {f01['recall']:.3f} |",
        f"| Recall at 1% FPR | {f1['recall']:.3f} |",
        f"| Brier score | {m['brier']:.6f} |",
        f"| Expected calibration error | {m['ece']:.6f} |",
        "",
        "Recall per typology:",
        "",
        "| Typology | at 0.1% FPR | at 1% FPR |",
        "|----------|-------------|-----------|",
    ]
    for t in schema.TYPOLOGIES[1:]:
        if t in f1:
            lines.append(f"| {t} | {f01.get(t, 0):.3f} | {f1[t]:.3f} |")
    lines += [
        "",
        "## Cost",
        "",
        f"Missed fraud costs its amount, a missed money-mule or structuring case a flat",
        f"{c['aml_miss_cost_eur']:.0f} EUR, each alert {c['review_cost_eur']:.0f} EUR of analyst time. The",
        f"threshold {c['threshold']:.4f} minimises that cost on the calibration window; on the",
        f"test window it raises {c['alerts']} alerts and costs {c['model_eur']:.0f} EUR against",
        f"{c['no_model_eur']:.0f} EUR with no model.",
        "",
        "## Reason codes and importance",
        "",
        "Reason codes are per-feature contributions to the log-odds: for trees, the change",
        "in a node's expected value at each split on the path, credited to the split's",
        "feature (bias plus contributions equals the margin exactly); for the logistic",
        "model, weight times standardised value. Global importance (share of split gain",
        "or of |weight|):",
        "",
        "| Feature | Share |",
        "|---------|-------|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in top]
    lines += [
        "",
        "## Drift",
        "",
        "Population stability index between the training and test windows. Above 0.2:",
        "",
    ]
    lines += [f"- {k}: {v:.3f}" for k, v in drift.items()] or ["- none"]
    lines += [
        "",
        f"Score PSI: {m['drift_psi']['score']:.4f}. In production the OCaml `Monitor` keeps",
        "the same histogram over live scores and exports it for alerting.",
        "",
        "## Limitations",
        "",
        "- Synthetic data: the typologies are the generator's, so the metrics measure how",
        "  well the features separate them, not real-world performance.",
        "- Authorised push payment scams look like genuine customer activity and stay the",
        "  hardest typology; rules (confirmation of payee, cooling-off for new",
        "  beneficiaries) carry most of that load.",
        "- Features need the customer's recent history; a cold-start customer scores on",
        "  defaults.",
        "- Balances in the generator grow over the window (salaries exceed spending), so",
        "  `balance_drain_ratio` drifts between training and test; the drift check is",
        "  expected to flag it.",
        "",
        "## Feature schema",
        "",
        f"`{schema.SCHEMA_NAME}` v{schema.SCHEMA_VERSION}, digest `{schema.digest()}`.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data/synthetic.npz")
    p.add_argument("--trained", default="data/trained.pkl")
    p.add_argument("--models", default="../models")
    a = p.parse_args()
    d = load(a.data)
    # The pickle is the local output of train.py, never a downloaded file.
    with open(a.trained, "rb") as f:
        trained = pickle.load(f)
    for name, entry in trained["models"].items():
        m = evaluate(name, entry, d)
        out = os.path.join(a.models, name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(m, f, indent=1)
        with open(os.path.join(out, "CARD.md"), "w") as f:
            f.write(card(m, entry))
        f1 = m["at_fpr_1%"]
        print(f"{name}: PR-AUC {m['pr_auc']:.4f} ROC-AUC {m['roc_auc']:.4f} "
              f"recall@0.1% {m['at_fpr_0.1%']['recall']:.3f} recall@1% {f1['recall']:.3f} "
              f"cost {m['cost']['model_eur']:.0f}/{m['cost']['no_model_eur']:.0f} EUR")
        print("   per typology @1%: " + ", ".join(
            f"{t} {f1[t]:.2f}" for t in schema.TYPOLOGIES[1:] if t in f1))
        print(f"   score PSI {m['drift_psi']['score']:.4f}; drifted features: "
              + (", ".join(k for k, v in m["drift_psi"].items() if v > 0.2) or "none"))


if __name__ == "__main__":
    main()
