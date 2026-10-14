# AGENTS.md: docs/

`api-notes.md`: notes on the scikit-learn and LightGBM internals the export relies on (tree arrays,
child ordering, leaf values, missing-value handling), checked against pinned versions.

- Read it before changing `python/export.py` or `lib/lightgbm_dump.ml`.
- Keep the versions in it current when a dependency in `python/uv.lock` changes.
