open! Core
module Rr = Risk_rules

let to_risk_rules model =
  let module M = struct
    let name = Model.name model
    let version = Model.version model

    let score features =
      let s = Model.explain model (Encoding.encode features) in
      Rr.Model.Score.create
        ~value:s.score
        ~reasons:(List.map s.reasons ~f:(fun r -> r.feature, r.contribution))
      |> Or_error.ok_exn
    ;;
  end
  in
  (module M : Rr.Model.S)
;;

let review_thresholds model =
  Option.map (Model.thresholds model) ~f:(fun t ->
    { Rr.Rules.Thresholds.challenge_at = None
    ; review_at = Some t.review
    ; block_at = None
    })
;;

let block_thresholds model =
  Option.bind (Model.thresholds model) ~f:(fun t ->
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
    let spec = Rr.Rule_spec.Score { model = Model.name model; thresholds } in
    if shadow then Rr.Rule_spec.Shadow spec else spec)
;;
