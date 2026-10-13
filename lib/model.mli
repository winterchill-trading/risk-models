(** A scoring model loaded from the JSON export written by [python/export.py].

    Loading checks the file's feature schema digest against the schema the caller builds
    vectors for, so a model trained on another layout fails at startup rather than scoring
    garbage. Scoring is [margin -> sigmoid -> calibration]; the [_u] functions return
    unboxed floats and allocate nothing. *)

open! Core

module Body : sig
  type t =
    | Forest of Forest.t
    | Linear of Linear.t
  [@@deriving sexp_of]
end

module Thresholds : sig
  (** Calibrated-score cut-offs for the two model rules (see [Model_rules]): [review]
      keeps alerts within the daily budget, [block] is the lowest score whose validation
      precision reached the target. *)
  type t =
    { review : float
    ; block : float
    ; alerts_per_day : int (** the budget [review] was chosen for *)
    ; block_precision : float (** the precision target [block] was chosen for *)
    }
  [@@deriving sexp, compare, equal]
end

type t [@@deriving sexp_of]

val create
  :  ?thresholds:Thresholds.t
  -> ?reference:float array * int array
  -> name:string
  -> schema:Schema.t
  -> Body.t
  -> Calibration.t
  -> t Or_error.t

(** [expected] defaults to [Schema.txn_features_v1]. *)
val of_yojson : ?expected:Schema.t -> Yojson.Safe.t -> t Or_error.t

val load : ?expected:Schema.t -> string -> t Or_error.t
val name : t -> string
val schema : t -> Schema.t
val body : t -> Body.t
val calibration : t -> Calibration.t
val thresholds : t -> Thresholds.t option

(** Score histogram of the calibration window: bucket edges (n + 1) and counts (n). *)
val reference : t -> (float array * int array) option

val margin_u : t -> float array -> float# [@@zero_alloc]
val probability_u : t -> float array -> float# [@@zero_alloc]

(** Calibrated probability. *)
val score_u : t -> float array -> float# [@@zero_alloc]

val margin : t -> float array -> float
val probability : t -> float array -> float
val score : t -> float array -> float

(** Writes per-feature contributions to the margin into [into] and returns the margin. For
    trees these are path attributions, for the linear model the weighted terms. *)
val contributions_u : t -> float array -> into:float array -> float#
[@@zero_alloc]

(** Score with the top [k] (default 3) reasons. Allocates. *)
val explain : ?k:int -> t -> float array -> Score.t
