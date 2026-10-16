# AGENTS.md: bin/

The `risk-models` command (`main.ml`): `schema`, `encode`, `check`, `score`, `explain`, `bench`.

- `schema` writes the feature schema as JSON; the Python side reads that output.
- `encode` turns risk-rules feature lines into CSV rows in model order.
- `check` loads a model, verifies the schema digest and prints a summary.
- `explain` prints one Explanation JSON line (format 1) per row given with `-features` or `-csv`;
  `-background` switches to interventional attributions.
- `score` and `bench` run the scorer over CSV rows and report latency and allocation.
- Output formats are read by scripts and by the Python tests. Changing a column is a breaking change.
- No environment variables, credentials or network access are used. Keep it that way.
