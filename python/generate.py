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
# Inbound sender ids: the employer, a handful of friends and family, and random
# one-off payers drawn from a large range.
EMPLOYER, FRIENDS = 0, 1000
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
            c.events.append(Event(t, "inbound", amount, benef=int(rng.integers(2**20, 2**30))))
            t += float(rng.uniform(0.5, 20)) * HOUR
        t += float(rng.uniform(1, 24)) * HOUR
        c.events.append(Event(t, "transfer", round(total * float(rng.uniform(0.5, 1.0)), 2),
                              benef=c.new_benef(t)))
    for day in range(c.days):
        dow = day % 7
        abroad = trip is not None and trip[0] <= day < trip[1]
        device = upgrade_device if upgrade is not None and day * DAY > upgrade else 0
        if day % 30 == c.salary_day % 30:
            c.events.append(Event(day * DAY + 7 * HOUR, "inbound", c.salary, benef=EMPLOYER))
        if rng.random() < 0.05:
            c.events.append(Event(pick_time(rng, day, TRANSFER_HOURS), "inbound",
                                  float(rng.lognormal(math.log(80), 1.0)),
                                  benef=FRIENDS + int(rng.integers(0, 8))))
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
                                  round(float(rng.uniform(9000, 30000)), 2),
                                  benef=FRIENDS + int(rng.integers(0, 20))))
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
            c.events.append(Event(t, "inbound", amount, benef=int(rng.integers(2**20, 2**30))))
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
    c.events.append(Event(t, "inbound", round(n * float(rng.uniform(8000, 10000)), 2),
                          benef=FRIENDS + 100))
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


NAN = float("nan")
MCC_COMMON = [5411, 5812, 5814, 5541, 4111, 5912, 5651, 5999, 4121, 7832, 5311, 5200]
# Electronics, gift cards, gambling, quasi-cash and crypto, money transfer, direct marketing.
MCC_RISKY = [5732, 5947, 7995, 6051, 4829, 5967]
MCC_ATM = 6011
FRAUD_TRANSFER = {2, 3, 5}


def features(c, rng):
    """Walks the customer's events in time order and emits one row per outgoing event,
    computed only from what happened strictly before it, with the field meanings of
    risk-rules `Features` (see schema.py). Card payments are not part of the outgoing
    history counts and sums, exactly as there: those cover transfers only."""
    rows, labels, times, amounts = [], [], [], []
    c.events.sort(key=lambda e: e.t)
    balance = c.salary * float(rng.uniform(0.5, 3))
    out_1h, out_24h, out_7d, out_30d = deque(), deque(), deque(), deque()
    inbound_24h, card_10m, failed_sca = deque(), deque(), deque()
    sum_1h = sum_24h = sum_7d = sum_30d = in_sum = 0.0
    last_inbound = None
    payee_check = {}
    benef_first_used, benef_foreign = {}, {}
    # Established beneficiaries: about half were confirmed as trusted.
    benef_trusted = {b: rng.random() < 0.5 for b in range(c.n_benef)}
    # Devices first seen before the window are already known.
    devices_used = {(c.id, d) for d, day in c.devices.items() if day < 0}
    cred_change = -float(rng.uniform(10, 1500)) * DAY if rng.random() < 0.4 else None
    for e in c.events:
        t = e.t
        if e.kind == "inbound":
            balance += e.amount
            inbound_24h.append((t, e.amount, e.benef))
            in_sum += e.amount
            last_inbound = t
            continue
        if e.kind == "cred_change":
            cred_change = t
            continue
        if e.amount < 0:  # takeover or savings move: a share of the balance
            e.amount = round(max(balance, 50.0) * -e.amount, 2)
        while out_1h and out_1h[0][0] <= t - HOUR:
            sum_1h -= out_1h.popleft()[1]
        while out_24h and out_24h[0][0] <= t - DAY:
            sum_24h -= out_24h.popleft()[1]
        while out_7d and out_7d[0][0] <= t - 7 * DAY:
            sum_7d -= out_7d.popleft()[1]
        while out_30d and out_30d[0][0] <= t - 30 * DAY:
            sum_30d -= out_30d.popleft()[1]
        while inbound_24h and inbound_24h[0][0] <= t - DAY:
            in_sum -= inbound_24h.popleft()[1]
        while card_10m and card_10m[0][0] <= t - 600:
            card_10m.popleft()
        while failed_sca and failed_sca[0] <= t - DAY:
            failed_sca.popleft()
        # Card testing and stolen-card bursts come with declined attempts; a takeover
        # often fails strong authentication before getting through. Both happen just
        # before the scored event and are not scored themselves.
        if e.kind == "card" and rng.random() < (0.3 if e.label == 1 else 0.01):
            for _ in range(int(rng.integers(1, 4))):
                card_10m.append((t - float(rng.uniform(5, 300)), False))
        if e.kind == "transfer" and rng.random() < (0.35 if e.label == 2 else 0.01):
            for _ in range(int(rng.integers(1, 4))):
                failed_sca.append(t - float(rng.uniform(60, 3 * HOUR)))
        # A mule account may exist only from part-way through the window.
        if c.account_age0 + t / DAY < 0:
            continue
        is_card = e.kind == "card"
        fraud = e.label > 0
        check = None
        rail = card_online = card_atm = None
        mcc = NAN
        benef_age = trusted = counterparty_home = NAN
        self_hosted = 0.0
        if is_card:
            atm = not fraud and not e.online and rng.random() < 0.08
            if atm:
                e.amount = float(max(20, 20 * round(e.amount * 2 / 20)))
            card_online, card_atm = e.online, atm
            mcc = float(MCC_ATM if atm else
                        MCC_RISKY[rng.integers(len(MCC_RISKY))] if e.mcc_risk else
                        MCC_COMMON[rng.integers(len(MCC_COMMON))])
            counterparty_home = 0.0 if e.foreign else 1.0
        else:
            if e.crypto:
                rail = "onchain" if rng.random() < 0.7 else "lightning"
                self_hosted = float(rng.random() < (0.7 if fraud else 0.15))
            elif e.label in FRAUD_TRANSFER:
                rail = "sepa_instant" if rng.random() < 0.8 else "sepa_credit"
            else:
                u = rng.random()
                rail = "sepa_instant" if u < 0.4 else "intraledger" if u < 0.45 else "sepa_credit"
            if e.benef not in benef_foreign:
                benef_foreign[e.benef] = rng.random() < (0.35 if fraud else 0.05)
            if not e.crypto:
                counterparty_home = 0.0 if benef_foreign[e.benef] else 1.0
            first = benef_first_used.get(e.benef, c.benef_added[e.benef] * DAY)
            benef_age = max(0.0, (t - first) / HOUR)
            trusted = 1.0 if benef_trusted.get(e.benef, False) else 0.0
            if e.benef not in payee_check:
                # Verification of payee: scam payees are often named differently from
                # what the victim was told; established payees matched long ago.
                if e.benef < c.n_benef:
                    p = [1.0, 0.0, 0.0, 0.0]
                elif e.label == 5:
                    p = [0.4, 0.25, 0.3, 0.05]
                elif e.label in (2, 3):
                    p = [0.6, 0.15, 0.15, 0.1]
                else:
                    p = [0.85, 0.08, 0.04, 0.03]
                payee_check[e.benef] = ["Match", "Close_match", "No_match", "Not_possible"][
                    rng.choice(4, p=p)]
            check = None if e.crypto else payee_check[e.benef]
            round_p = 0.4 if e.label == 4 else 0.3 if e.label == 3 else 0.15 if not fraud else 0.05
            if e.amount >= 100 and rng.random() < round_p:
                e.amount = float(100 * max(1, round(e.amount / 100)))
        amount = e.amount
        bal = balance if rng.random() > 0.01 else NAN  # the balance read can fail
        ip_home = NAN if rng.random() < 0.02 else (0.0 if e.ip_mismatch else 1.0)
        dev = (c.id, e.device)
        row = {
            "is_card": float(is_card),
            "amount": amount,
            "amount_is_round": float(amount > 0 and amount % 100 == 0),
            "balance": bal,
            "amount_to_balance": amount / bal if bal == bal and bal > 0 else NAN,
            "beneficiary_age_hours": benef_age,
            "beneficiary_trusted": trusted,
            "counterparty_country_is_residence": counterparty_home,
            "self_hosted_unverified": self_hosted,
            "mcc": mcc,
            "out_count_1h": float(len(out_1h)),
            "out_count_7d": float(len(out_7d)),
            "out_sum_1h": max(sum_1h, 0.0),
            "out_sum_7d": max(sum_7d, 0.0),
            "amount_over_mean_30d":
                amount / (sum_30d / len(out_30d)) if out_30d and sum_30d > 1e-9 else NAN,
            "in_distinct_senders_24h": float(len({b for _, _, b in inbound_24h})),
            "minutes_since_last_inbound":
                (t - last_inbound) / 60 if last_inbound is not None else NAN,
            "failed_sca_24h": float(len(failed_sca)),
            "card_auths_10m": float(len(card_10m)) if is_card else 0.0,
            "card_declines_10m":
                float(sum(1 for _, ok in card_10m if not ok)) if is_card else 0.0,
            "log1p_amount": math.log1p(amount),
            "is_weekend": float(int(t // DAY) % 7 >= 5),
            "payee_match": float(check == "Match"),
            "payee_close_match": float(check == "Close_match"),
            "payee_no_match": float(check == "No_match"),
            "payee_not_possible": float(check == "Not_possible"),
            "out_count_24h": float(len(out_24h)),
            "out_sum_24h": max(sum_24h, 0.0),
            "out_sum_30d": max(sum_30d, 0.0),
            "out_distinct_beneficiaries_24h": float(len({b for _, _, b in out_24h})),
            "in_count_24h": float(len(inbound_24h)),
            "in_sum_24h": max(in_sum, 0.0),
            "pass_through_ratio_24h":
                (max(sum_24h, 0.0) + amount) / in_sum if in_sum > 1e-9 else NAN,
            "new_device": float(dev not in devices_used),
            "ip_country_is_residence": ip_home,
            "hours_since_credentials_change":
                (t - cred_change) / HOUR if cred_change is not None else NAN,
            "account_age_days": max(0.0, c.account_age0 + t / DAY),
            "hour_of_day_utc": float(int((t % DAY) // HOUR)),
            "rail_sepa_instant": float(rail == "sepa_instant"),
            "rail_onchain": float(rail == "onchain"),
            "rail_lightning": float(rail == "lightning"),
            "rail_intraledger": float(rail == "intraledger"),
            "card_online": float(bool(card_online)),
            "card_atm": float(bool(card_atm)),
        }
        rows.append([row[f] for f in schema.FEATURES])
        labels.append(e.label)
        times.append(t)
        amounts.append(amount)
        devices_used.add(dev)
        balance -= amount
        if is_card:
            card_10m.append((t, True))
        else:
            benef_first_used.setdefault(e.benef, t)
            out_1h.append((t, amount))
            out_24h.append((t, amount, e.benef))
            out_7d.append((t, amount))
            out_30d.append((t, amount))
            sum_1h += amount
            sum_24h += amount
            sum_7d += amount
            sum_30d += amount
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
