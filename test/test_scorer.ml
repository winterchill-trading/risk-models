open! Core
open Risk_models

let gbdt = lazy (Scorer.load "../models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn)

let%expect_test "the bundled model loads with its thresholds and reference histogram" =
  let m = force gbdt in
  let trees, nodes =
    match Scorer.body m with
    | Forest f -> Forest.n_trees f, Forest.n_nodes f
    | Linear _ -> 0, 0
  in
  print_s
    [%message
      (Scorer.name m)
        (trees : int)
        (nodes : int)
        ~thresholds:(Scorer.thresholds m : Scorer.Thresholds.t option)
        ~reference_buckets:
          (Option.map (Scorer.reference m) ~f:(fun (_, c) -> Array.length c) : int option)];
  [%expect
    {|
    (txn-fraud-gbdt (trees 150) (nodes 4350)
     (thresholds
      (((review 0.11538461538461539) (block (0.68888888888888888))
        (alerts_per_day 50) (block_precision 0.9))))
     (reference_buckets (11)))
    |}]
;;

let%expect_test "a model for another schema is refused" =
  let other =
    Schema.create_exn ~name:"txn-features" ~version:2 [ "amount_log10"; "hour_sin" ]
  in
  (match Scorer.load ~expected:other "../models/txn-fraud-gbdt/model.json" with
   | Ok _ -> print_endline "loaded"
   | Error e -> print_s [%sexp (e : Error.t)]);
  [%expect
    {|
    (("loading model" ../models/txn-fraud-gbdt/model.json)
     ("feature schema mismatch"
      (expected (txn-features 2 7801ce9e8882eade968a767ac65949cc))
      (got (risk-rules-features 1 405b88b3692809d1dde4e6e214006b69))))
    |}]
;;

let%expect_test "malformed files name the failing field" =
  let show s =
    match Scorer.of_yojson (Yojson.Safe.from_string s) with
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
  let x = Array.create ~len:(Schema.length (Scorer.schema m)) 0. in
  let set f v = x.(Schema.index_exn (Scorer.schema m) f) <- v in
  (* An account-takeover shape: fresh credentials and device, a new payee, most of the
     balance, out by instant transfer. *)
  set "amount" 2400.;
  set "log1p_amount" (Float.log1p 2400.);
  set "balance" 2600.;
  set "amount_to_balance" (2400. /. 2600.);
  set "beneficiary_age_hours" 0.2;
  set "new_device" 1.;
  set "hours_since_credentials_change" 1.5;
  set "failed_sca_24h" 2.;
  set "account_age_days" 900.;
  set "hour_of_day_utc" 3.;
  set "rail_sepa_instant" 1.;
  set "payee_no_match" 1.;
  let s = Scorer.explain m x in
  print_s [%sexp (s : Risk_rules.Model.Score.t)];
  [%expect
    {|
    ((value 1)
     (reasons
      ((beneficiary_age_hours 9.3556574735409) (in_sum_24h 3.3774508440132225)
       (hour_of_day_utc 2.2342008693066453))))
    |}]
;;
