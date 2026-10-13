open! Core
open Risk_models

let gbdt = lazy (Model.load "../models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn)

let%expect_test "the bundled model loads with its thresholds and reference histogram" =
  let m = force gbdt in
  let trees, nodes =
    match Model.body m with
    | Forest f -> Forest.n_trees f, Forest.n_nodes f
    | Linear _ -> 0, 0
  in
  print_s
    [%message
      (Model.name m)
        (trees : int)
        (nodes : int)
        ~thresholds:(Model.thresholds m : Model.Thresholds.t option)
        ~reference_buckets:
          (Option.map (Model.reference m) ~f:(fun (_, c) -> Array.length c) : int option)];
  [%expect
    {|
    (txn-fraud-gbdt (trees 150) (nodes 4350)
     (thresholds
      (((review 0.14468085106382977) (block (0.8)) (alerts_per_day 50)
        (block_precision 0.9))))
     (reference_buckets (11)))
    |}]
;;

let%expect_test "a model for another schema is refused" =
  let other =
    Schema.create_exn ~name:"txn-features" ~version:2 [ "amount_log10"; "hour_sin" ]
  in
  (match Model.load ~expected:other "../models/txn-fraud-gbdt/model.json" with
   | Ok _ -> print_endline "loaded"
   | Error e -> print_s [%sexp (e : Error.t)]);
  [%expect
    {|
    (("loading model" ../models/txn-fraud-gbdt/model.json)
     ("feature schema mismatch"
      (expected (txn-features 2 7801ce9e8882eade968a767ac65949cc))
      (got (txn-features 1 4451fa83bbfdc7994f1f4942758d1968))))
    |}]
;;

let%expect_test "malformed files name the failing field" =
  let show s =
    match Model.of_yojson (Yojson.Safe.from_string s) with
    | Ok _ -> print_endline "ok"
    | Error e -> print_s [%sexp (e : Error.t)]
  in
  show {|{"format": 2}|};
  show {|{"format": 1, "name": "m"}|};
  [%expect
    {|
    ("unsupported model format" (format 2))
    (("in field" schema) ("missing field" schema))
    |}]
;;

let%expect_test "explain gives the score with its top reasons" =
  let m = force gbdt in
  let x = Array.create ~len:(Schema.length (Model.schema m)) 0. in
  let set f v = x.(Schema.index_exn (Model.schema m) f) <- v in
  set "amount_log10" 3.6;
  set "beneficiary_new" 1.;
  set "new_device" 1.;
  set "hours_since_credential_change_log" 1.;
  set "balance_drain_ratio" 0.9;
  set "account_age_days_log" 6.;
  set "hour_cos" 1.;
  let s = Model.explain m x in
  print_s [%sexp (s : Score.t)];
  [%expect
    {|
    ((model txn-fraud-gbdt) (score 1) (margin 5.24094141700435)
     (reasons
      (((feature amount_log10) (contribution 6.3287174011466281))
       ((feature beneficiary_age_days_log) (contribution 1.8865593426218867))
       ((feature hours_since_credential_change_log)
        (contribution 1.6860506695347846)))))
    |}]
;;
