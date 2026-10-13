"""Evaluate trained models on the held-out window and write each model card.

Splits are time-ordered with a 7-day label delay gap (risk_models_train/splits.py).

Metrics (on calibrated scores, held-out test window after the gap):
  average precision (PR-AUC) and ROC-AUC
  precision and recall at a daily alert budget (top k per day, k = 25, 50, 100)
  recall at 0.1% and 1% false-positive rate, overall and per typology
  Brier score and expected calibration error (10 equal-count bins)
  cost: a threshold chosen on the calibration window to minimise
        missed loss + review cost, then applied to the test window
  rules: the Review threshold that keeps the calibration window within 50 alerts a day
        and the Block threshold, the lowest score with precision >= 0.9 there, then
        both applied to the test window
  prequential: 4 weekly folds, each trained on everything before a 7-day gap
        (gradient-boosted model only), average precision and budget precision
  drift: population stability index of every feature and of the score between the
        training and test windows (review at 0.1, retrain at 0.25)
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
from risk_models_train.splits import DAY, prequential
from train import linear_inputs, load, make_gbdt, split

BUDGETS = [25, 50, 100]  # alerts a day an operations team can review
REVIEW_BUDGET = 50
BLOCK_PRECISION = 0.9
PSI_REVIEW, PSI_RETRAIN = 0.1, 0.25

REVIEW_COST = 4.0  # EUR per alert an analyst looks at
# Money-laundering typologies have no direct loss to the bank; a missed case is priced
# as a flat regulatory exposure instead of the amount.
AML_MISS_COST = 500.0
AML = {schema.TYPOLOGIES.index("money_mule"), schema.TYPOLOGIES.index("structuring")}


def raw_scores(entry, X):
    if "gbdt" in entry:
        return entry["gbdt"].predict_proba(X)[:, 1]
    if "lgbm" in entry:
        return entry["lgbm"].predict_proba(X)[:, 1]
    return entry["logit"].predict_proba(linear_view(entry, X))[:, 1]


def linear_view(entry, X):
    return entry["scaler"].transform(entry["imputer"].transform(linear_inputs(X)))


def scores(entry, X):
    return entry["iso"].predict(raw_scores(entry, X))


def at_budget(rank, y, day, k):
    """Each day, alert on the k highest-ranked rows. Precision is averaged over days,
    recall is over all positives."""
    tp, alerts = 0, 0
    precisions = []
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        top = idx[np.argsort(-rank[idx], kind="stable")[:k]]
        hits = int(y[top].sum())
        tp += hits
        alerts += len(top)
        precisions.append(hits / max(len(top), 1))
    return {"k": k, "precision": float(np.mean(precisions)),
            "recall": float(tp / max(int(y.sum()), 1))}


def rule_thresholds(s, y, days):
    """Review: the lowest calibrated score whose alerts stay within the daily budget.
    Block: the lowest score at which precision is still >= BLOCK_PRECISION (None when
    no score reaches it). Calibrated scores are piecewise constant, so both are chosen
    among the scores that occur."""
    values = np.unique(s)[::-1]
    order = np.argsort(-s, kind="stable")
    sorted_s, sorted_y = s[order], y[order]
    review, block = float(values[0]), None
    for v in values:
        n = int(np.searchsorted(-sorted_s, -v, side="right"))
        if n / days <= REVIEW_BUDGET:
            review = float(v)
        if n >= 10 and sorted_y[:n].mean() >= BLOCK_PRECISION:
            block = float(v)
    return review, block


def apply_rules(s, y, days, review, block):
    out = {}
    for name, th in [("review", review), ("block", block)]:
        if th is None:
            out[name] = None
            continue
        hit = s >= th
        out[name] = {"threshold": th, "alerts_per_day": float(hit.sum() / days),
                     "precision": float(y[hit].mean()) if hit.any() else 0.0,
                     "recall": float(hit[y == 1].mean())}
    return out


def prequential_report(d):
    X, y, day = d["X"], d["y"], np.floor(d["t"] / DAY)
    folds = []
    for k, tr, te in prequential(d["t"]):
        m = make_gbdt().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        folds.append({"fold": k, "test_days": [int(day[te].min()), int(day[te].max())],
                      "positives": int(y[te].sum()),
                      "average_precision": float(average_precision_score(y[te], p)),
                      **{f"precision_at_{b}": at_budget(p, y[te], day[te], b)["precision"]
                         for b in BUDGETS}})
    ap = [f["average_precision"] for f in folds]
    return {"folds": folds, "average_precision_mean": float(np.mean(ap)),
            "average_precision_std": float(np.std(ap))}


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
    if "lgbm" in entry:
        gain = entry["lgbm"].booster_.feature_importance(importance_type="gain")
        return gain / gain.sum()
    w = np.abs(entry["logit"].coef_[0])
    return w / w.sum()


def evaluate(name, entry, d):
    X, y, typ, amount = d["X"], d["y"], d["typology"], d["amount"]
    tr, ca, te = split(d["t"])
    s_te, s_ca, s_tr = scores(entry, X[te]), scores(entry, X[ca]), scores(entry, X[tr])
    raw_te = raw_scores(entry, X[te])
    day = np.floor(d["t"] / DAY)
    ca_days, te_days = len(np.unique(day[ca])), len(np.unique(day[te]))
    th = best_threshold(s_ca, typ[ca], amount[ca])
    review, block = rule_thresholds(s_ca, y[ca], ca_days)
    m = {
        "name": name,
        "rows": {"train": int(tr.sum()), "calibration": int(ca.sum()), "test": int(te.sum())},
        "positives_test": int(y[te].sum()),
        "pr_auc": float(average_precision_score(y[te], s_te)),
        "roc_auc": float(roc_auc_score(y[te], s_te)),
        "brier": float(brier_score_loss(y[te], s_te)),
        "ece": ece(s_te, y[te]),
        # Ranked by the uncalibrated probability: calibration is monotone but piecewise
        # constant, and ties would make "top k" ambiguous.
        "at_budget": [at_budget(raw_te, y[te], day[te], b) for b in BUDGETS],
        "rules": {"review_budget_per_day": REVIEW_BUDGET, "block_precision": BLOCK_PRECISION,
                  "review": review, "block": block,
                  "calibration_window": apply_rules(s_ca, y[ca], ca_days, review, block),
                  "test_window": apply_rules(s_te, y[te], te_days, review, block)},
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


def fmt_rule(r):
    if r is None:
        return "| - | never fires | - | - |"
    return (f"| {r['threshold']:.4f} | {r['alerts_per_day']:.1f} | {r['precision']:.3f} "
            f"| {r['recall']:.3f} |")


def card(m, entry):
    kind = ("gradient-boosted trees (scikit-learn HistGradientBoosting)" if "gbdt" in entry
            else "gradient-boosted trees (LightGBM)" if "lgbm" in entry
            else "logistic regression on standardised features")
    f1, f01 = m["at_fpr_1%"], m["at_fpr_0.1%"]
    top = sorted(m["importance"].items(), key=lambda kv: -kv[1])[:8]
    c, r = m["cost"], m["rules"]
    lines = [
        f"# {m['name']}",
        "",
        f"Generated by `python/evaluate.py` on {datetime.date.today().isoformat()}.",
        "",
        "> **Synthetic data only.** Every number on this page comes from data the",
        "> generator in `python/generate.py` made up, with typologies it injected itself.",
        "> The metrics say how well the features separate those made-up patterns. They say",
        "> nothing about performance on real payments; retrain, re-evaluate and re-choose",
        "> every threshold on real, labelled traffic before relying on the model.",
        "",
        "## What it is",
        "",
        f"A {kind} scoring one outgoing payment (card authorisation or transfer) for",
        "fraud and money-laundering typologies. The output is a calibrated probability",
        "(isotonic regression fitted on a later time window than training) with up to",
        "three reason codes: the features that pushed the score up the most.",
        "",
        "## Intended use: two rules, shadow mode first",
        "",
        "The model's output enters the decision as two rules next to the deterministic",
        "rules in risk-rules:",
        "",
        f"- `model_review` (severity Review) when the calibrated score is at least",
        f"  {r['review']:.4f}, the lowest score that kept the calibration window within",
        f"  {r['review_budget_per_day']} alerts a day (the daily alert budget).",
        "- `model_block` (severity Block) when the score is at least the lowest value whose",
        f"  precision on the calibration window was {r['block_precision']:.0%} or more"
        + (f": {r['block']:.4f}." if r["block"] is not None else
           "; no score reached that precision, so this rule is disabled."),
        "",
        "Both rules start in shadow mode (severity Allow): hits are journaled with the",
        "score, the model name and the top three reasons, but change no decision. Promote",
        "them after two weeks of shadow running with score PSI below 0.1 and alert counts",
        "close to the budget. The thresholds ship in `model.json` and are only a starting",
        "point.",
        "",
        "| Rule | Threshold | Alerts a day | Precision | Recall |",
        "|------|-----------|--------------|-----------|--------|",
        "| review (calibration window) " + fmt_rule(r["calibration_window"]["review"]),
        "| review (test window) " + fmt_rule(r["test_window"]["review"]),
        "| block (calibration window) " + fmt_rule(r["calibration_window"]["block"]),
        "| block (test window) " + fmt_rule(r["test_window"]["block"]),
        "",
        "## Data and validation",
        "",
        "Synthetic only (`python/generate.py`). No public transaction-monitoring dataset",
        "is downloadable without registration (PaySim and the card-fraud sets need a",
        "Kaggle login), so the generator models customers with daily and weekly seasonality",
        "and injects five typologies: card-not-present bursts, account takeover, money",
        "mules, structuring under the 10 000 EUR threshold and authorised push payment",
        "fraud.",
        "",
        "Splits are time-ordered, never random, with a 7-day label delay gap: days 0-51",
        "train, 52-65 calibrate, 66-72 are dropped (labels such as chargebacks and scam",
        "claims arrive late, so the last week before a decision point is unlabelled), 73",
        f"onwards test. Rows: {m['rows']['train']} train, {m['rows']['calibration']}",
        f"calibration, {m['rows']['test']} test ({m['positives_test']} positive in test).",
        "",
        "## Metrics (test window)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average precision (PR-AUC) | {m['pr_auc']:.4f} |",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| Recall at 0.1% FPR | {f01['recall']:.3f} |",
        f"| Recall at 1% FPR | {f1['recall']:.3f} |",
        f"| Brier score | {m['brier']:.6f} |",
        f"| Expected calibration error | {m['ece']:.6f} |",
        "",
        "At a daily alert budget (the k highest scores each day):",
        "",
        "| Alerts a day | Precision | Recall |",
        "|--------------|-----------|--------|",
    ]
    lines += [f"| {b['k']} | {b['precision']:.3f} | {b['recall']:.3f} |" for b in m["at_budget"]]
    if "prequential" in m:
        pq = m["prequential"]
        lines += [
            "",
            "Prequential check: four weekly test folds, newest first, each trained on",
            "everything before a 7-day gap:",
            "",
            "| Fold | Test days | Positives | Average precision | Precision at 50/day |",
            "|------|-----------|-----------|-------------------|---------------------|",
        ]
        lines += [f"| {f['fold']} | {f['test_days'][0]}-{f['test_days'][1]} | {f['positives']} "
                  f"| {f['average_precision']:.4f} | {f['precision_at_50']:.3f} |"
                  for f in pq["folds"]]
        lines += ["", f"Average precision {pq['average_precision_mean']:.4f} "
                      f"+/- {pq['average_precision_std']:.4f} across folds."]
    lines += [
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
        "Missed fraud costs its amount, a missed money-mule or structuring case a flat",
        f"{c['aml_miss_cost_eur']:.0f} EUR, each alert {c['review_cost_eur']:.0f} EUR of analyst time. The",
        f"threshold {c['threshold']:.4f} minimises that cost on the calibration window; on the",
        f"test window it raises {c['alerts']} alerts and costs {c['model_eur']:.0f} EUR against",
        f"{c['no_model_eur']:.0f} EUR with no model.",
        "",
        "## Reason codes and importance",
        "",
        "Reason codes are per-feature contributions to the log-odds: for trees, the change",
        "in a node's expected value at each split on the path, credited to the split's",
        "feature (bias plus contributions equals the margin exactly; cheaper than",
        "TreeSHAP but order-dependent); for the logistic model, weight times standardised",
        "value. Global importance (share of split gain or of |weight|):",
        "",
        "| Feature | Share |",
        "|---------|-------|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in top]
    watch = {k: v for k, v in m["drift_psi"].items() if v >= PSI_REVIEW}
    lines += [
        "",
        "## Drift",
        "",
        f"Population stability index between the training and test windows; review at",
        f"{PSI_REVIEW}, retrain at {PSI_RETRAIN}. At or above {PSI_REVIEW}:",
        "",
    ]
    lines += [f"- {k}: {v:.3f}" + (" (retrain)" if v >= PSI_RETRAIN else " (review)")
              for k, v in watch.items()] or ["- none"]
    lines += [
        "",
        f"Score PSI: {m['drift_psi']['score']:.4f}. In production the OCaml `Monitor` keeps",
        "the same score histogram over live traffic and exports it for alerting.",
        "",
        "## Limitations",
        "",
        "- Synthetic data: the typologies are the generator's, so the metrics measure how",
        "  well the features separate them, not real-world performance.",
        "- Authorised push payment fraud looks like genuine customer activity and stays",
        "  hard; rules (verification of payee, cooling-off for new beneficiaries) carry",
        "  much of that load.",
        "- Features need the customer's recent history; a cold-start customer scores on",
        "  defaults.",
        "- Rolling 30-day sums start empty on the generator's first day, so",
        "  `out_sum_30d` is lower in the early training days than in the test window; the",
        "  drift check is expected to flag it.",
        "- Card velocity comes only from `card_auths_10m` and `card_declines_10m`; the",
        "  outgoing counts and sums in risk-rules `Features` cover transfers, so stolen-card",
        "  bursts stay the weakest typology.",
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
        if "gbdt" in entry:
            m["prequential"] = prequential_report(d)
        out = os.path.join(a.models, name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(m, f, indent=1)
            f.write("\n")
        with open(os.path.join(out, "CARD.md"), "w") as f:
            f.write(card(m, entry))
        f1, r = m["at_fpr_1%"], m["rules"]["test_window"]
        b50 = next(b for b in m["at_budget"] if b["k"] == 50)
        print(f"{name}: AP {m['pr_auc']:.4f} ROC-AUC {m['roc_auc']:.4f} "
              f"recall@0.1%FPR {m['at_fpr_0.1%']['recall']:.3f} recall@1%FPR {f1['recall']:.3f} "
              f"precision@50/day {b50['precision']:.3f} recall@50/day {b50['recall']:.3f}")
        print("   per typology @1%FPR: " + ", ".join(
            f"{t} {f1[t]:.2f}" for t in schema.TYPOLOGIES[1:] if t in f1))
        for k in ("review", "block"):
            print(f"   {k} rule on test: {r[k]}")
        if "prequential" in m:
            pq = m["prequential"]
            print(f"   prequential AP {pq['average_precision_mean']:.4f} "
                  f"+/- {pq['average_precision_std']:.4f}")
        print(f"   score PSI {m['drift_psi']['score']:.4f}; PSI >= {PSI_REVIEW}: "
              + (", ".join(k for k, v in m["drift_psi"].items() if v >= PSI_REVIEW) or "none"))


if __name__ == "__main__":
    main()
