open! Core
open Risk_models

let%expect_test "digest matches the Python schema" =
  let s = Schema.txn_features_v1 in
  print_s [%message (s.digest : string) (Schema.length s : int)];
  [%expect {| ((s.digest 4) ("Schema.length s" 25)) |}]
;;

let%expect_test "create rejects empty and duplicate names" =
  let show r = print_s [%sexp (r : Schema.t Or_error.t)] in
  show (Schema.create ~name:"x" ~version:1 []);
  show (Schema.create ~name:"x" ~version:1 [ "a"; "b"; "a" ]);
  [%expect {| |}]
;;

let%expect_test "compatibility is by digest" =
  let a = Schema.create_exn ~name:"a" ~version:1 [ "x"; "y" ] in
  let b = Schema.create_exn ~name:"b" ~version:2 [ "x"; "y" ] in
  let c = Schema.create_exn ~name:"a" ~version:1 [ "y"; "x" ] in
  print_s [%sexp (Schema.check_compatible ~expected:a b : unit Or_error.t)];
  print_s [%sexp (Schema.check_compatible ~expected:a c : unit Or_error.t)];
  [%expect {| |}]
;;

let%expect_test "stable sexp round trip re-derives the digest" =
  let s = Schema.txn_features_v1 in
  let back = Schema.Stable.V1.(t_of_sexp (sexp_of_t s)) in
  print_s [%message (Schema.equal s back : bool) (Schema.index s "new_device" : int option)];
  [%expect {| |}]
;;
