# AGENTS.md: python/

The training pipeline. It writes the models that `../models/` commits and the parity fixtures in
`../test/fixtures/`. Python 3.12 or newer, managed with `uv`.

Setup and runs, from this directory:

```sh
uv sync --locked                       # install the pinned dependencies from uv.lock
uv run python generate.py              # data/synthetic.npz (synthetic, 6000 customers, 91 days)
uv run python train.py                 # data/trained.pkl: gradient-boosted trees and a logistic baseline
uv run python evaluate.py              # ../models/<name>/metrics.json and CARD.md
uv run python export.py                # ../models/<name>/model.json, ../test/fixtures/*.csv (parity
                                       # and shap attribution fixtures)
```

- Use `uv sync --locked` (not a plain `uv sync`) so the lock file is respected. Change dependencies with
  `uv add` and commit `uv.lock` in the same commit.
- `data/` is gitignored; generated data never goes in a commit.
- `risk_models_train/features.json` is written by the OCaml CLI (`risk-models schema`), not by hand.
- Splits are by time (`risk_models_train/splits.py`). Keep the day ranges in `train.py`'s header in
  step with that module.
- The nightly workflow runs this same sequence and uploads its output. Do not
  copy its output into `../models/` without a person's review.
