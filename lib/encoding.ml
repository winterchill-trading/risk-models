open! Core
module Features = Risk_rules.Features

(* Each one-hot group has an all-zeros baseline: SEPA credit transfers and cards for the
   rail, in-person for the card channel, no payee check (cards, crypto). *)
let one_hot : (string * (Features.t -> bool)) list =
  [ ("rail_sepa_instant", fun f -> String.equal f.rail "sepa_instant")
  ; ("rail_onchain", fun f -> String.is_prefix f.rail ~prefix:"onchain:")
  ; ("rail_lightning", fun f -> String.equal f.rail "lightning")
  ; ("rail_intraledger", fun f -> String.equal f.rail "intraledger")
  ; ("card_online", fun f -> Option.equal String.equal f.card_channel (Some "Online"))
  ; ("card_atm", fun f -> Option.equal String.equal f.card_channel (Some "Atm"))
  ; ("payee_match", fun f -> Option.equal String.equal f.payee_check (Some "Match"))
  ; ( "payee_close_match"
    , fun f -> Option.equal String.equal f.payee_check (Some "Close_match") )
  ; ("payee_no_match", fun f -> Option.equal String.equal f.payee_check (Some "No_match"))
  ; ( "payee_not_possible"
    , fun f -> Option.equal String.equal f.payee_check (Some "Not_possible") )
  ]
;;

let one_hot_names = List.map one_hot ~f:fst

let schema =
  Schema.create_exn
    ~name:"risk-rules-features"
    ~version:Features.schema_version
    (Features.numeric_names @ one_hot_names)
;;

let n_numeric = List.length Features.numeric_names

let encode_into (f : Features.t) into =
  if Array.length into <> Schema.length schema
  then invalid_arg "Encoding.encode_into: wrong vector length";
  List.iteri (Features.to_floats f) ~f:(fun i (_, v) -> into.(i) <- v);
  List.iteri one_hot ~f:(fun i (_, p) -> into.(n_numeric + i) <- (if p f then 1. else 0.))
;;

let encode f =
  let into = Array.create ~len:(Schema.length schema) 0. in
  encode_into f into;
  into
;;

let schema_json () =
  `Assoc
    [ "name", `String schema.name
    ; "version", `Int schema.version
    ; "features", `List (Array.to_list schema.features |> List.map ~f:(fun f -> `String f))
    ; "digest", `String schema.digest
    ]
;;
