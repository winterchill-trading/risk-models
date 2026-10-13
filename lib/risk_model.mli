(** A risk-models [Model.t] as risk-rules sees it.

    risk-rules scores a {!Risk_rules.Features.t} through [Risk_rules.Model.S]; this module
    encodes the features ({!Encoding}), scores them and passes the calibrated score and
    the top reasons on. The model's output then becomes two rules:

    - [model_review] (Review) when the score reaches the [review] threshold, chosen to
      keep alerts within the daily budget;
    - [model_block] (Block) when it reaches [block], the lowest score with the target
      precision on the calibration window (absent when no score reached it).

    Both default to shadow mode (capped at Allow): hits are journaled with the score,
    model name, version and reasons, but change no decision until promoted. *)

open! Core

(** [Risk_rules.Model.S] over [model]. Reasons are the top three features by contribution
    to the log-odds. *)
val to_risk_rules : Model.t -> Risk_rules.Model.t

(** The model file's thresholds as risk-rules thresholds, one per rule. *)
val review_thresholds : Model.t -> Risk_rules.Rules.Thresholds.t option

val block_thresholds : Model.t -> Risk_rules.Rules.Thresholds.t option

(** [model_review] and [model_block] (when the model has the thresholds), capped at Allow
    unless [shadow] is [false]. *)
val rules : ?shadow:bool -> Model.t -> Risk_rules.Context.t Risk_rules.Rule.t list

(** The same two rules as policy data ([Score] specs, wrapped in [Shadow] unless [shadow]
    is [false]), for a policy sexp file that names the model. *)
val rule_specs : ?shadow:bool -> Model.t -> Risk_rules.Rule_spec.t list
