# AGENTS.md: .github/

Two workflows.

- `ci.yml`: the OCaml build and tests on the committed models and fixtures, with the `ox-setup` pins
  (fixed-decimal, money, ledger, payments-core, risk-rules) that must match `risk-models.opam`.
- `nightly.yml`: the full Python pipeline on a schedule, in `python/`, with an average-precision floor.
  It never replaces committed models.
- Check runs with `gh run list --limit 1`; at most one `gh run watch <id> --exit-status`. No loops.
