# AGENTS.md: test/

Inline tests (ppx_expect, Quickcheck) for the schema, encoding, forest, calibration, scorer, monitor,
risk-model adapter and stable formats.

- `test_parity.ml` scores the fixtures in `fixtures/` and compares with the Python scores. The fixtures
  come from `python/export.py`. Regenerate them there, never by hand.
- `test_stable.ml` pins the Bin_prot shape digests of the types that are journaled. A changed digest
  means journaled scores or calibration tables change format: decide that on purpose and say so in the commit.
- The test stanza depends on `../models` and `../python/risk_models_train/features.json`. A missing
  file fails the build, which is intended.
- Promote expect output with `dune promote` after reading the diff.
