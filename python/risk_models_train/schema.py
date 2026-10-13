"""Feature schema shared by the generator, the trainer and the exporter.

The order of FEATURES is the column order of every feature vector, in Python and in
OCaml. The digest is the hex MD5 of the names joined with newlines; a model file
records it and the OCaml loader refuses a model whose digest differs from the schema
it scores against.
"""

import hashlib
import json
import os

# The feature list comes from the OCaml side: `risk-models schema` prints risk-rules
# `Features.numeric_names` (the vector `Features.to_floats` builds: booleans 0/1, a
# missing option NaN) followed by the one-hot columns risk-models adds for the
# categorical fields (`rail`, `card_channel`, `payee_check`). risk-rules owns the
# meaning of every field. Regenerate features.json after a risk-rules schema change;
# an OCaml test fails while the committed file and the library disagree.
with open(os.path.join(os.path.dirname(__file__), "features.json")) as _f:
    _SCHEMA = json.load(_f)

SCHEMA_NAME = _SCHEMA["name"]
SCHEMA_VERSION = _SCHEMA["version"]
FEATURES = _SCHEMA["features"]

# Heavy-tailed features the logistic baseline reads as log1p(max(x, 0)); the trees do
# not care about monotone transforms and read the raw values.
LOG_FEATURES = [
    "amount", "balance", "amount_to_balance", "beneficiary_age_hours", "out_sum_1h",
    "out_sum_24h", "out_sum_7d", "out_sum_30d", "in_sum_24h", "pass_through_ratio_24h",
    "hours_since_credentials_change", "account_age_days", "amount_over_mean_30d",
    "minutes_since_last_inbound",
]

# Label names follow the risk-rules Typology names where one exists.
TYPOLOGIES = ["legit", "cnp_burst", "account_takeover", "money_mule", "structuring", "app_fraud"]


def digest(names=None):
    names = FEATURES if names is None else names
    return hashlib.md5("\n".join(names).encode()).hexdigest()


assert digest() == _SCHEMA["digest"], "features.json digest does not match its names"
