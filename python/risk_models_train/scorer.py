"""Reference scorer over the exported JSON model.

This is the specification the OCaml `Model` module follows: same traversal, same
summation order, same link and calibration. export.py checks it against scikit-learn
before writing a model, and writes the parity fixture from it.
"""

import bisect
import math


def margin_and_contributions(model, x):
    """Margin (log-odds before calibration) and per-feature contributions to it.

    Trees: contributions are path attributions (the change in the node's expected value
    at each split, credited to the split's feature), so bias + sum(contributions) equals
    the margin. Logistic: weight times standardised value, relative to the training mean.
    """
    n = len(model["schema"]["features"])
    contrib = [0.0] * n
    if model["kind"] == "gbdt":
        g = model["gbdt"]
        margin = g["base_score"]
        bias = g["base_score"]
        for tree in g["trees"]:
            node = 0
            bias += tree["value"][0]
            while tree["feature"][node] >= 0:
                f = tree["feature"][node]
                v = x[f]
                if math.isnan(v):
                    child = tree["left"][node] if tree["missing_left"][node] else tree["right"][node]
                elif v <= tree["threshold"][node]:
                    child = tree["left"][node]
                else:
                    child = tree["right"][node]
                contrib[f] += tree["value"][child] - tree["value"][node]
                node = child
            margin += tree["value"][node]
        return margin, contrib, bias
    if model["kind"] == "lightgbm":
        return lightgbm_margin(model["lightgbm"], x, contrib)
    lin = model["logistic"]
    margin = lin["intercept"]
    for i, (w, m, s, lg) in enumerate(
            zip(lin["weights"], lin["mean"], lin["scale"], lin["log1p"])):
        v = x[i]
        if math.isnan(v):
            continue  # missing: imputed with the training mean, so the term is zero
        if lg:
            v = math.log1p(max(v, 0.0))
        c = w * ((v - m) / s)
        contrib[i] = c
        margin += c
    return margin, contrib, lin["intercept"]


def _count(node):
    if "leaf_value" in node:
        return float(node.get("leaf_count", 1))
    return float(node["internal_count"])


def _expected(node):
    """Count-weighted mean of the leaves below, same formula as lib/lightgbm_dump.ml."""
    if "leaf_value" in node:
        return node["leaf_value"]
    l, r = node["left_child"], node["right_child"]
    cl, cr = _count(l), _count(r)
    return (cl * _expected(l) + cr * _expected(r)) / (cl + cr)


def _goes_left(node, v):
    """LightGBM's NumericalDecision for double inputs."""
    missing = node["missing_type"]
    if math.isnan(v) and missing != "NaN":
        v = 0.0
    if (missing == "Zero" and -1e-35 <= v <= 1e-35) or (missing == "NaN" and math.isnan(v)):
        return node["default_left"]
    return v <= node["threshold"]


def lightgbm_margin(dump, x, contrib):
    margin = 0.0
    bias = 0.0
    for tree in dump["tree_info"]:
        node = tree["tree_structure"]
        bias += _expected(node)
        while "leaf_value" not in node:
            child = node["left_child"] if _goes_left(node, x[node["split_feature"]]) \
                else node["right_child"]
            contrib[node["split_feature"]] += _expected(child) - _expected(node)
            node = child
        margin += node["leaf_value"]
    return margin, contrib, bias


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def calibrate(cal, p):
    if cal["kind"] == "identity":
        return p
    if cal["kind"] == "platt":
        return sigmoid(cal["a"] * p + cal["b"])
    xs, ys = cal["x"], cal["y"]
    # numpy.interp semantics with clipping at both ends.
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    j = bisect.bisect_right(xs, p) - 1
    slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
    return slope * (p - xs[j]) + ys[j]


def reasons(contrib, k=3):
    """Indices of the k features that pushed the score up the most, largest first."""
    pos = [(c, i) for i, c in enumerate(contrib) if c > 0]
    pos.sort(key=lambda ci: (-ci[0], ci[1]))
    return [i for _, i in pos[:k]]


def score(model, x):
    margin, contrib, _ = margin_and_contributions(model, x)
    p = sigmoid(margin)
    return margin, p, calibrate(model["calibration"], p), reasons(contrib)
