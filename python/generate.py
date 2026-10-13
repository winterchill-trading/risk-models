"""Synthetic retail-banking activity with injected fraud and money-laundering typologies.

Each customer has an account age, a spending scale, card and transfer rates, a set of
known beneficiaries, a salary and a few devices. Legitimate activity follows a daily
and weekly rhythm (card spend peaks at lunch and early evening, transfers on weekday
mornings, less on Sundays). Legitimate noise is deliberately confusable with fraud:
trips abroad, phone upgrades, new beneficiaries, one-off large purchases.

Injected typologies (label = index in schema.TYPOLOGIES):
  cnp_burst         stolen card details used online: 3-15 purchases within an hour or
                    two, often at night, high-risk merchant categories, foreign merchants
  account_takeover  credential change and a new device, then transfers draining the
                    balance to new beneficiaries or crypto within hours
  money_mule        young account receives several inbound credits and forwards most of
                    them within a day to new beneficiaries or crypto
  structuring       repeated transfers just under the 10 000 EUR reporting threshold
  app_fraud         the genuine customer, on their own device, sends unusually large
                    transfers to a brand new beneficiary (authorised push payment scam)

Output: data/<name>.npz with X (features in schema order), y (0/1), typology, t
(seconds from the start), customer and amount.

Usage: uv run python generate.py [--customers 6000] [--days 91] [--seed 7] [--out data/synthetic.npz]
"""

import argparse
import math
import os
from collections import deque

import numpy as np

from risk_models_train import schema

DAY = 86400.0
HOUR = 3600.0
THRESHOLD = 10000.0

# Hour-of-day weights (index = hour). Card spend has lunch and evening peaks; transfers
# cluster in the working morning.
CARD_HOURS = np.array(
    [1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 3, 4, 5, 6, 9, 8, 6, 5, 6, 8, 10, 9, 7, 5, 3, 2],
    dtype=float,
)
TRANSFER_HOURS = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 2, 5, 8, 9, 8, 6, 5, 6, 6, 5, 4, 3, 3, 2, 1.5, 1, 0.5],
    dtype=float,
)
NIGHT_HOURS = np.array([6, 6, 6, 5, 5, 4] + [1] * 16 + [3, 5], dtype=float)
# Monday = 0. Card spend rises on Friday and Saturday; transfers drop at the weekend.
CARD_DOW = np.array([0.9, 0.9, 0.95, 1.0, 1.2, 1.3, 0.8])
TRANSFER_DOW = np.array([1.3, 1.1, 1.0, 1.0, 1.1, 0.4, 0.2])


def norm(w):
    return w / w.sum()


CARD_HOURS, TRANSFER_HOURS, NIGHT_HOURS = map(norm, (CARD_HOURS, TRANSFER_HOURS, NIGHT_HOURS))


class Event:
    __slots__ = (
        "t", "kind", "amount", "benef", "online", "crypto", "foreign", "mcc_risk",
        "ip_mismatch", "device", "label",
    )

    def __init__(self, t, kind, amount, *, benef=-1, online=False, crypto=False,
                 foreign=False, mcc_risk=False, ip_mismatch=False, device=0, label=0):
        self.t = t
        self.kind = kind  # "card", "transfer", "inbound", "cred_change"
        self.amount = amount
        self.benef = benef
        self.online = online
        self.crypto = crypto
        self.foreign = foreign
        self.mcc_risk = mcc_risk
        self.ip_mismatch = ip_mismatch
        self.device = device
        self.label = label


class Customer:
    def __init__(self, rng, cid, days):
        self.id = cid
        self.days = days
        self.account_age0 = float(rng.lognormal(math.log(700), 1.0))
        self.card_scale = float(rng.lognormal(math.log(35), 0.5))
        self.transfer_scale = float(rng.lognormal(math.log(250), 0.7))
        self.card_rate = float(rng.gamma(2.0, 0.6))
        self.transfer_rate = float(rng.gamma(1.5, 0.1))
        self.online_share = float(rng.beta(2, 4))
        self.crypto_user = rng.random() < 0.08
        self.salary = float(rng.lognormal(math.log(2600), 0.4))
        self.salary_day = int(rng.integers(0, 28))
        self.n_benef = int(rng.integers(2, 9))
        # Beneficiary id -> age in days at the start of the window (negative = added later).
        self.benef_added = {b: -float(rng.uniform(30, 900)) for b in range(self.n_benef)}
        self.next_benef = self.n_benef
        self.devices = {0: -float(rng.uniform(30, 600))}
        self.next_device = 1
        self.events = []

    def new_benef(self, t):
        b = self.next_benef
        self.next_benef += 1
        self.benef_added[b] = t / DAY
        return b

    def new_device(self, t):
        d = self.next_device
        self.next_device += 1
        self.devices[d] = t / DAY
        return d


def pick_time(rng, day, hours):
    hour = rng.choice(24, p=hours)
    return day * DAY + hour * HOUR + float(rng.uniform(0, HOUR))


def legit(rng, c):
    trip = None
    if rng.random() < 0.15:
        start = int(rng.integers(0, c.days - 10))
        trip = (start, start + int(rng.integers(3, 11)))
    # Phone upgrades and password resets: a credential change and a new device, the
    # first half of an account takeover's footprint.
    upgrade, upgrade_device = None, 0
    if rng.random() < 0.10:
        upgrade = float(rng.uniform(0, c.days)) * DAY
        c.events.append(Event(upgrade, "cred_change", 0.0))
        upgrade_device = c.new_device(upgrade)
    # Moving savings to an account at another bank: a large share of the balance to a
    # new beneficiary, sometimes right after the upgrade.
    if rng.random() < 0.04:
        t = (upgrade + float(rng.uniform(1, 72)) * HOUR) if upgrade is not None and \
            rng.random() < 0.3 else pick_time(rng, int(rng.integers(1, c.days)), TRANSFER_HOURS)
        if t < c.days * DAY:
            c.events.append(Event(t, "transfer", -float(rng.uniform(0.3, 0.9)),
                                  benef=c.new_benef(t - float(rng.uniform(0, 48)) * HOUR),
                                  crypto=c.crypto_user and rng.random() < 0.5,
                                  device=upgrade_device if upgrade is not None and t > upgrade
                                  else 0))
    # Online shopping sprees (sales, gifts, concert tickets), at any hour for night owls.
    for _ in range(rng.poisson(0.25)):
        day = int(rng.integers(0, c.days))
        t = pick_time(rng, day, NIGHT_HOURS if rng.random() < 0.25 else CARD_HOURS)
        for _ in range(int(rng.integers(3, 9))):
            t += float(rng.uniform(60, 1500))
            c.events.append(Event(
                t, "card", round(float(rng.lognormal(math.log(c.card_scale * 2), 0.8)), 2),
                online=True, foreign=rng.random() < 0.3, mcc_risk=rng.random() < 0.3,
                ip_mismatch=rng.random() < 0.03))
    # Small-business owners paying invoices near the reporting line.
    business = rng.random() < 0.012
    # Selling a car or splitting a group trip: several inbound credits, then a payout.
    if rng.random() < 0.02:
        day = int(rng.integers(1, c.days - 2))
        t = pick_time(rng, day, TRANSFER_HOURS)
        total = 0.0
        for _ in range(int(rng.integers(2, 6))):
            amount = round(float(rng.uniform(200, 2500)), 2)
            total += amount
            c.events.append(Event(t, "inbound", amount))
            t += float(rng.uniform(0.5, 20)) * HOUR
        t += float(rng.uniform(1, 24)) * HOUR
        c.events.append(Event(t, "transfer", round(total * float(rng.uniform(0.5, 1.0)), 2),
                              benef=c.new_benef(t)))
    for day in range(c.days):
        dow = day % 7
        abroad = trip is not None and trip[0] <= day < trip[1]
        device = upgrade_device if upgrade is not None and day * DAY > upgrade else 0
        if day % 30 == c.salary_day % 30:
            c.events.append(Event(day * DAY + 7 * HOUR, "inbound", c.salary))
        if rng.random() < 0.05:
            c.events.append(Event(pick_time(rng, day, TRANSFER_HOURS), "inbound",
                                  float(rng.lognormal(math.log(80), 1.0))))
        for _ in range(rng.poisson(c.card_rate * CARD_DOW[dow])):
            online = rng.random() < c.online_share
            amount = float(rng.lognormal(math.log(c.card_scale), 0.8))
            if rng.random() < 0.006:
                amount *= float(rng.uniform(8, 30))  # a sofa, a laptop
            c.events.append(Event(
                pick_time(rng, day, CARD_HOURS), "card", round(amount, 2), online=online,
                foreign=abroad or (online and rng.random() < 0.15),
                mcc_risk=rng.random() < (0.15 if online else 0.04),
                ip_mismatch=abroad or rng.random() < 0.03, device=device))
        if business and dow < 5 and rng.random() < 0.15:
            c.events.append(Event(day * DAY + 7 * HOUR, "inbound",
                                  round(float(rng.uniform(9000, 30000)), 2)))
            t = pick_time(rng, day, TRANSFER_HOURS)
            benef = c.new_benef(t) if rng.random() < 0.3 else int(rng.integers(0, c.n_benef))
            c.events.append(Event(t, "transfer", round(float(rng.uniform(4000, 9999)), 2),
                                  benef=benef))
        for _ in range(rng.poisson(c.transfer_rate * TRANSFER_DOW[dow])):
            t = pick_time(rng, day, TRANSFER_HOURS)
            crypto = c.crypto_user and rng.random() < 0.3
            if rng.random() < 0.08 or crypto and rng.random() < 0.2:
                benef = c.new_benef(t - float(rng.uniform(0, 72)) * HOUR)
                big = rng.random() < 0.15  # a deposit, a holiday rental, a used car
            else:
                benef = int(rng.integers(0, c.n_benef))
                big = rng.random() < 0.01
            amount = float(rng.lognormal(math.log(c.transfer_scale), 0.9))
            if big:
                amount *= float(rng.uniform(3, 25))
            c.events.append(Event(t, "transfer", round(min(amount, 25000.0), 2), benef=benef,
                                  crypto=crypto, ip_mismatch=abroad, device=device))


def cnp_burst(rng, c):
    day = int(rng.integers(5, c.days))
    t = pick_time(rng, day, NIGHT_HOURS if rng.random() < 0.5 else CARD_HOURS)
    foreign = rng.random() < 0.5
    for _ in range(int(rng.integers(2, 12))):
        t += float(rng.uniform(60, 2400))
        c.events.append(Event(
            t, "card", round(float(rng.lognormal(math.log(90), 0.9)), 2), online=True,
            foreign=foreign or rng.random() < 0.2, mcc_risk=rng.random() < 0.5,
            ip_mismatch=rng.random() < 0.35, label=1))


def account_takeover(rng, c):
    day = int(rng.integers(10, c.days))
    t = pick_time(rng, day, NIGHT_HOURS if rng.random() < 0.3 else TRANSFER_HOURS)
    # Remote-access takeovers run on the victim's own device and need no reset.
    remote = rng.random() < 0.25
    device = 0
    if not remote:
        c.events.append(Event(t, "cred_change", 0.0))
        device = c.new_device(t)
    ip = rng.random() < (0.2 if remote else 0.6)
    for _ in range(int(rng.integers(1, 4))):
        t += float(rng.uniform(0.2, 30)) * HOUR
        crypto = rng.random() < 0.4
        # Amount is set from the balance when features are computed (share of balance).
        c.events.append(Event(t, "transfer", -float(rng.uniform(0.25, 0.9)),
                              benef=c.new_benef(t), crypto=crypto, ip_mismatch=ip,
                              device=device, label=2))


def mule(rng, c):
    # Mule accounts are mostly opened recently and recruited to pass money through.
    c.account_age0 = float(rng.uniform(-20, 60)) if rng.random() < 0.7 \
        else float(rng.uniform(60, 1500))
    for _ in range(int(rng.integers(1, 4))):
        day = int(rng.integers(max(0, int(-c.account_age0)) + 2, c.days - 2))
        t = pick_time(rng, day, TRANSFER_HOURS)
        total = 0.0
        for _ in range(int(rng.integers(1, 6))):
            amount = round(float(rng.uniform(300, 3000)), 2)
            total += amount
            c.events.append(Event(t, "inbound", amount))
            t += float(rng.uniform(0.2, 12)) * HOUR
        out = total * float(rng.uniform(0.7, 0.95))
        parts = int(rng.integers(1, 3))
        benef = c.new_benef(t) if rng.random() < 0.8 else int(rng.integers(0, c.n_benef))
        for _ in range(parts):
            t += float(rng.uniform(0.5, 30)) * HOUR
            c.events.append(Event(t, "transfer", round(out / parts, 2), benef=benef,
                                  crypto=rng.random() < 0.3, label=3))
            if rng.random() < 0.5:
                benef = c.new_benef(t)


def structuring(rng, c):
    day = int(rng.integers(3, c.days - 20))
    t = pick_time(rng, day, TRANSFER_HOURS)
    n = int(rng.integers(2, 7))
    c.events.append(Event(t, "inbound", round(n * float(rng.uniform(8000, 10000)), 2)))
    for _ in range(n):
        t += float(rng.uniform(6, 120)) * HOUR
        benef = c.new_benef(t) if rng.random() < 0.4 else int(rng.integers(0, c.n_benef))
        c.events.append(Event(t, "transfer", round(float(rng.uniform(0.7, 0.995)) * THRESHOLD, 2),
                              benef=benef, label=4))


def app_scam(rng, c):
    day = int(rng.integers(5, c.days))
    t = pick_time(rng, day, TRANSFER_HOURS)
    # Scammers have the victim add the payee first, sometimes days ahead.
    benef = c.new_benef(t - float(rng.uniform(0, 96)) * HOUR)
    for _ in range(int(rng.integers(1, 4))):
        amount = c.transfer_scale * float(rng.uniform(1.5, 20))
        c.events.append(Event(t, "transfer", round(min(amount, 20000.0), 2), benef=benef,
                              label=5))
        t += float(rng.uniform(0.5, 72)) * HOUR


TYPOLOGY_FNS = [(cnp_burst, 0.03), (account_takeover, 0.03), (mule, 0.02),
                (structuring, 0.015), (app_scam, 0.03)]


def features(c, rng):
    """Walks the customer's events in time order and emits one row per outgoing event,
    computed only from what happened strictly before it."""
    rows, labels, times, amounts = [], [], [], []
    c.events.sort(key=lambda e: e.t)
    balance = c.salary * float(rng.uniform(0.5, 3))
    out_1h, out_24h, out_30d_card, out_30d_tr = deque(), deque(), deque(), deque()
    inbound_24h, benef_24h, near_7d = deque(), deque(), deque()
    sum_24h = 0.0
    in_sum = 0.0
    paid = set()
    cred_change = None
    for e in c.events:
        t = e.t
        if e.kind == "inbound":
            balance += e.amount
            inbound_24h.append((t, e.amount))
            in_sum += e.amount
            continue
        if e.kind == "cred_change":
            cred_change = t
            continue
        if e.amount < 0:  # takeover: a share of the balance
            e.amount = round(max(balance, 50.0) * -e.amount, 2)
        while out_1h and out_1h[0] <= t - HOUR:
            out_1h.popleft()
        while out_24h and out_24h[0][0] <= t - DAY:
            sum_24h -= out_24h.popleft()[1]
        for q in (out_30d_card, out_30d_tr):
            while q and q[0][0] <= t - 30 * DAY:
                q.popleft()
        while inbound_24h and inbound_24h[0][0] <= t - DAY:
            in_sum -= inbound_24h.popleft()[1]
        while benef_24h and benef_24h[0][0] <= t - DAY:
            benef_24h.popleft()
        while near_7d and near_7d[0] <= t - 7 * DAY:
            near_7d.popleft()
        is_card = e.kind == "card"
        hist = out_30d_card if is_card else out_30d_tr
        avg = sum(a for _, a in hist) / len(hist) if hist else e.amount
        hour = (t % DAY) / HOUR
        dow = int(t // DAY) % 7
        if is_card:
            benef_new, benef_age = 0.0, 0.0
        else:
            benef_new = 0.0 if e.benef in paid else 1.0
            benef_age = math.log1p(max(0.0, t / DAY - c.benef_added[e.benef]))
        device_age = t / DAY - c.devices.get(e.device, -1000.0)
        hours_cred = (t - cred_change) / HOUR if cred_change is not None else 24 * 365.0
        near = 1.0 if 0.85 * THRESHOLD <= e.amount < THRESHOLD else 0.0
        row = [
            math.log10(1 + e.amount),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if dow >= 5 else 0.0,
            1.0 if is_card else 0.0,
            1.0 if e.online else 0.0,
            1.0 if e.crypto else 0.0,
            benef_new,
            benef_age,
            float(len(out_1h)),
            float(len(out_24h)),
            math.log10(1 + max(sum_24h, 0.0)),
            max(-3.0, min(3.0, math.log10(max(e.amount, 0.01) / max(avg, 0.01)))),
            float(len({b for _, b in benef_24h})),
            math.log10(1 + max(in_sum, 0.0)),
            min(1.0, max(in_sum, 0.0) / (max(balance, 0.0) + 1)),
            near,
            float(len(near_7d)),
            1.0 if device_age < 1.0 else 0.0,
            math.log1p(min(hours_cred, 24 * 365.0)),
            1.0 if e.ip_mismatch else 0.0,
            1.0 if e.foreign else 0.0,
            1.0 if e.mcc_risk else 0.0,
            math.log1p(max(0.0, c.account_age0 + t / DAY)),
            min(5.0, e.amount / (max(balance, 0.0) + 1)),
        ]
        # A mule account may exist only from part-way through the window.
        if c.account_age0 + t / DAY < 0:
            continue
        rows.append(row)
        labels.append(e.label)
        times.append(t)
        amounts.append(e.amount)
        balance -= e.amount
        out_1h.append(t)
        out_24h.append((t, e.amount))
        sum_24h += e.amount
        hist.append((t, e.amount))
        if not is_card:
            paid.add(e.benef)
            benef_24h.append((t, e.benef))
        if near:
            near_7d.append(t)
    return rows, labels, times, amounts


def generate(customers, days, seed):
    rng = np.random.default_rng(seed)
    X, typ, T, C, A = [], [], [], [], []
    for cid in range(customers):
        c = Customer(rng, cid, days)
        for fn, p in TYPOLOGY_FNS:
            if rng.random() < p:
                fn(rng, c)
        legit(rng, c)
        rows, labels, times, amounts = features(c, rng)
        X.extend(rows)
        typ.extend(labels)
        T.extend(times)
        C.extend([cid] * len(rows))
        A.extend(amounts)
    order = np.argsort(np.asarray(T), kind="stable")
    X = np.asarray(X, dtype=np.float64)[order]
    typ = np.asarray(typ, dtype=np.int8)[order]
    assert X.shape[1] == len(schema.FEATURES)
    return dict(X=X, y=(typ > 0).astype(np.int8), typology=typ,
                t=np.asarray(T)[order], customer=np.asarray(C, dtype=np.int32)[order],
                amount=np.asarray(A)[order])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--customers", type=int, default=6000)
    p.add_argument("--days", type=int, default=91)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/synthetic.npz")
    a = p.parse_args()
    data = generate(a.customers, a.days, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **data, features=np.array(schema.FEATURES),
                        digest=np.array(schema.digest()))
    n = len(data["y"])
    counts = np.bincount(data["typology"], minlength=len(schema.TYPOLOGIES))
    print(f"{n} events from {a.customers} customers over {a.days} days -> {a.out}")
    for name, k in zip(schema.TYPOLOGIES, counts):
        print(f"  {name:18s} {k:8d}  {100 * k / n:6.3f}%")


if __name__ == "__main__":
    main()
