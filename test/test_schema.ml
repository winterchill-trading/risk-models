open! Core
open Risk_models

let%expect_test "the committed features.json is the encoding's schema" =
  let s = Encoding.schema in
  let committed = Yojson.Safe.from_file "../python/risk_models_train/features.json" in
  print_s
    [%message
      (s.digest : string)
        (Schema.length s : int)
        ~same_as_committed:(Yojson.Safe.equal committed (Encoding.schema_json ()) : bool)];
  [%expect
    {|
    ((s.digest 405b88b3692809d1dde4e6e214006b69) ("Schema.length s" 44)
     (same_as_committed true))
    |}]
;;

let%expect_test "create rejects empty and duplicate names" =
  let show r = print_s [%sexp (r : Schema.t Or_error.t)] in
  show (Schema.create ~name:"x" ~version:1 []);
  show (Schema.create ~name:"x" ~version:1 [ "a"; "b"; "a" ]);
  [%expect
    {|
    (Error ("a feature schema needs at least one feature" x))
    (Error ("duplicate feature name" x a))
    |}]
;;

let%expect_test "compatibility is by digest" =
  let a = Schema.create_exn ~name:"a" ~version:1 [ "x"; "y" ] in
  let b = Schema.create_exn ~name:"b" ~version:2 [ "x"; "y" ] in
  let c = Schema.create_exn ~name:"a" ~version:1 [ "y"; "x" ] in
  print_s [%sexp (Schema.check_compatible ~expected:a b : unit Or_error.t)];
  print_s [%sexp (Schema.check_compatible ~expected:a c : unit Or_error.t)];
  [%expect
    {|
    (Ok ())
    (Error
     ("feature schema mismatch" (expected (a 1 16151ff14c884e2d18c9903202288ba0))
      (got (a 1 3519e47d48e8ba68c296508a0e6b6946))))
    |}]
;;

let%expect_test "stable sexp round trip re-derives the digest" =
  let s = Encoding.schema in
  let back = Schema.Stable.V1.(t_of_sexp (sexp_of_t s)) in
  print_s
    [%message (Schema.equal s back : bool) (Schema.index s "new_device" : int option)];
  [%expect {| (("Schema.equal s back" true) ("Schema.index s \"new_device\"" (18))) |}]
;;
