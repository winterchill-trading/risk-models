"""Feature schema shared by the generator, the trainer and the exporter.

The order of FEATURES is the column order of every feature vector, in Python and in
OCaml. The digest is the hex MD5 of the names joined with newlines; a model file
records it and the OCaml loader refuses a model whose digest differs from the schema
it scores against.
"""

import hashlib

SCHEMA_NAME = "txn-features"
SCHEMA_VERSION = 1

FEATURES = [
    "amount_log10",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_card",
    "is_online",
    "is_crypto",
    "beneficiary_new",
    "beneficiary_age_days_log",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h_log10",
    "amount_ratio_30d_log10",
    "distinct_beneficiaries_24h",
    "inbound_24h_log10",
    "inbound_share_of_balance",
    "near_threshold",
    "near_threshold_count_7d",
    "new_device",
    "hours_since_credential_change_log",
    "ip_country_mismatch",
    "foreign_merchant",
    "mcc_high_risk",
    "account_age_days_log",
    "balance_drain_ratio",
]

# Label names follow the risk-rules Typology names where one exists.
TYPOLOGIES = ["legit", "cnp_burst", "account_takeover", "money_mule", "structuring", "app_fraud"]


def digest(names=None):
    names = FEATURES if names is None else names
    return hashlib.md5("\n".join(names).encode()).hexdigest()
