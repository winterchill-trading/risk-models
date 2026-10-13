(lang dune 3.17)

(name risk-models)

(generate_opam_files true)

(license MIT)

(maintainers "winterchill-trading")

(authors "winterchill-trading")

(source
 (github winterchill-trading/risk-models))

(package
 (name risk-models)
 (allow_empty)
 (synopsis "Fraud and transaction monitoring models: Python training, OCaml scoring")
 (depends
  (ocaml
   (>= 5.2))
  dune
  core
  ppx_jane
  yojson))
