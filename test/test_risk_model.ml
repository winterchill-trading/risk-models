open! Core
open Risk_models
module Features = Risk_rules.Features

let model = lazy (Scorer.load "../models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn)

(* A takeover-shaped instant transfer: new device, credentials changed an hour and a half
   ago, a payee first used minutes ago whose name did not match, most of the balance. *)
let takeover : Features.t =
  { schema_version = Features.schema_version
  ; is_card = false
  ; rail = "sepa_instant"
  ; card_channel = None
  ; currency = "EUR"
  ; amount = 2400.
  ; amount_is_round = false
  ; balance = Some 2600.
  ; amount_to_balance = Some (2400. /. 2600.)
  ; beneficiary_age_hours = Some 0.2
  ; beneficiary_trusted = Some false
  ; counterparty_country_is_residence = Some false
  ; self_hosted_unverified = false
  ; mcc = None
  ; out_count_1h = 0
  ; out_count_24h = 0
  ; out_sum_24h = 0.
  ; out_sum_30d = 300.
  ; out_distinct_beneficiaries_24h = 0
  ; in_count_24h = 0
  ; in_sum_24h = 0.
  ; pass_through_ratio_24h = None
  ; new_device = true
  ; ip_country_is_residence = Some false
  ; hours_since_credentials_change = Some 1.5
  ; account_age_days = Some 900.
  ; hour_of_day_utc = 3
  ; log1p_amount = Float.log1p 2400.
  ; is_weekend = false
  ; out_count_7d = 1
  ; out_sum_1h = 0.
  ; out_sum_7d = 300.
  ; amount_over_mean_30d = Some 8.
  ; in_distinct_senders_24h = 0
  ; minutes_since_last_inbound = Some 5000.
  ; failed_sca_24h = 2
  ; payee_check = Some "No_match"
  ; card_auths_10m = 0
  ; card_declines_10m = 0
  }
;;

let groceries =
  { takeover with
    is_card = true
  ; rail = "card"
  ; card_channel = Some "In_person"
  ; amount = 42.1
  ; log1p_amount = Float.log1p 42.1
  ; amount_to_balance = Some (42.1 /. 2600.)
  ; beneficiary_age_hours = None
  ; beneficiary_trusted = None
  ; counterparty_country_is_residence = Some true
  ; mcc = Some 5411
  ; new_device = false
  ; ip_country_is_residence = Some true
  ; hours_since_credentials_change = None
  ; hour_of_day_utc = 12
  ; amount_over_mean_30d = Some 0.4
  ; failed_sca_24h = 0
  ; payee_check = None
  }
;;

let%expect_test "encoding lays out to_floats then the one-hot columns" =
  let x = Encoding.encode takeover in
  let show f = x.(Schema.index_exn Encoding.schema f) in
  print_s
    [%message
      ""
        ~n:(Array.length x : int)
        ~amount:(show "amount" : float)
        ~mcc:(show "mcc" : float)
        ~rail_sepa_instant:(show "rail_sepa_instant" : float)
        ~payee_no_match:(show "payee_no_match" : float)
        ~card_online:(show "card_online" : float)];
  [%expect
    {|
    ((n 44) (amount 2400) (mcc NAN) (rail_sepa_instant 1) (payee_no_match 1)
     (card_online 0))
    |}]
;;

let%expect_test "the model through risk-rules' Scorer.S" =
  let rr = Risk_model.to_risk_rules (force model) in
  List.iter
    [ "takeover", takeover; "groceries", groceries ]
    ~f:(fun (label, f) ->
      let s = Risk_rules.Model.score rr f in
      printf
        "%s: %.4f %s\n"
        label
        s.value
        (String.concat ~sep:", " (List.map s.reasons ~f:fst)));
  print_s
    [%message
      (Risk_rules.Model.name rr) (String.length (Risk_rules.Model.version rr) : int)];
  [%expect
    {|
    takeover: 1.0000 beneficiary_age_hours, in_sum_24h, hour_of_day_utc
    groceries: 0.0000 hours_since_credentials_change, amount_over_mean_30d, out_count_7d
    (txn-fraud-gbdt ("String.length (Risk_rules.Model.version rr)" 12))
    |}]
;;

let%expect_test "two rules, shadow by default, and their policy specs" =
  let m = force model in
  List.iter [ true; false ] ~f:(fun shadow ->
    List.iter (Risk_model.rules ~shadow m) ~f:(fun r ->
      printf
        "%s max %s\n"
        (Risk_rules.Rule.name r)
        (Sexp.to_string [%sexp (Risk_rules.Rule.severity r : Risk_rules.Severity.t)])));
  List.iter (Risk_model.rule_specs m) ~f:(fun s ->
    print_s [%sexp (s : Risk_rules.Rule_spec.t)]);
  [%expect
    {|
    model_review max Allow
    model_block max Allow
    model_review max Review
    model_block max Block
    (Shadow
     (Score (model txn-fraud-gbdt)
      (thresholds ((review_at 0.11538461538461539)))))
    (Shadow
     (Score (model txn-fraud-gbdt) (thresholds ((block_at 0.68888888888888888)))))
    |}]
;;

let%expect_test "risk-rules feature JSON (from `risk-rules features`) scores end to end" =
  let m = force model in
  In_channel.read_lines "fixtures/features-examples.jsonl"
  |> List.filter ~f:(fun l -> not (String.is_empty l))
  |> List.iter ~f:(fun line ->
    let f = Features.of_json line |> Or_error.ok_exn in
    let s = Scorer.explain m (Encoding.encode f) in
    printf
      "%s %s %.6f %s\n"
      f.rail
      (Option.value f.card_channel ~default:"-")
      s.value
      (String.concat ~sep:"," (List.map s.reasons ~f:fst)));
  [%expect
    {|
    card Online 0.002030 counterparty_country_is_residence,card_online,mcc
    sepa_instant - 0.023173 beneficiary_age_hours,payee_match,rail_sepa_instant
    |}]
;;
