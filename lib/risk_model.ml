open! Core
module Rr = Risk_rules

let to_risk_rules model =
  let module M = struct
    let name = Scorer.name model
    let version = Scorer.version model
    let score features = Scorer.explain model (Encoding.encode features)
  end
  in
  (module M : Rr.Model.S)
;;

let review_thresholds model =
  Option.map (Scorer.thresholds model) ~f:(fun t ->
    { Rr.Rules.Thresholds.challenge_at = None
    ; review_at = Some t.review
    ; block_at = None
    })
;;

let block_thresholds model =
  Option.bind (Scorer.thresholds model) ~f:(fun t ->
    Option.map t.block ~f:(fun block ->
      { Rr.Rules.Thresholds.challenge_at = None; review_at = None; block_at = Some block }))
;;

let named model =
  List.filter_opt
    [ Option.map (review_thresholds model) ~f:(fun t ->
        "model_review", Rr.Severity.Review, t)
    ; Option.map (block_thresholds model) ~f:(fun t ->
        "model_block", Rr.Severity.Block, t)
    ]
;;

let rules ?(shadow = true) model =
  let rr = to_risk_rules model in
  List.map (named model) ~f:(fun (name, severity, thresholds) ->
    Rr.Rules.model_score ~name rr thresholds
    |> Rr.Rule.cap ~max:(if shadow then Allow else severity))
;;

let rule_specs ?(shadow = true) model =
  List.map (named model) ~f:(fun (_, _, thresholds) ->
    let spec = Rr.Rule_spec.Score { model = Scorer.name model; thresholds } in
    if shadow then Rr.Rule_spec.Shadow spec else spec)
;;
