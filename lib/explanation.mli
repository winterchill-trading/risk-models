(** A compact, stable record of why a model scored one row as it did: the base value, the
    margin, the [k] features with the largest contributions (by absolute value) and the
    sum of the rest, so that [base + sum top + rest = margin] exactly as computed.

    Contributions are additive attributions to the log-odds ({!Tree_shap} for trees, the
    weighted terms for the logistic model). They say what the model leaned on for this
    row, not what caused the payment to be risky; label them "model contributions" in any
    screen a reviewer sees.

    Nothing here depends on a particular model: any caller with feature names, a row and
    an attribution vector can build one (a scorer embedding a {!Forest} of its own uses
    {!Tree_shap} and {!create} directly). *)

open! Core

module Method : sig
  type t =
    | Path_dependent (** trees, others follow the training covers *)
    | Interventional (** trees, others from a background sample *)
    | Linear (** logistic model: weighted standardised terms *)
  [@@deriving sexp, compare, equal, enumerate]

  (** ["path_dependent"], ["interventional"], ["linear"]. *)
  val to_string : t -> string

  val of_string : string -> t Or_error.t
end

module Contribution : sig
  type t =
    { feature : string (** schema name, e.g. ["beneficiary_age_hours"] *)
    ; label : string (** human-readable name; the schema name when none is given *)
    ; value : float option (** the row's value, [None] when missing *)
    ; contribution : float (** to the log-odds *)
    }
  [@@deriving sexp, compare, equal]
end

module Stable : sig
  module Method : sig
    module V1 : sig
      type t = Method.t [@@deriving bin_io, sexp, compare, equal]
    end
  end

  module Contribution : sig
    module V1 : sig
      type t = Contribution.t [@@deriving bin_io, sexp, compare, equal]
    end
  end

  module V1 : sig
    type t =
      { model : string
      ; version : string
      ; method_ : Method.V1.t
      ; base : float
      ; margin : float
      ; score : float option (** calibrated score, when the caller has one *)
      ; top : Contribution.V1.t list (** largest absolute contribution first *)
      ; rest : float (** sum of every other contribution *)
      }
    [@@deriving bin_io, sexp, compare, equal]
  end
end

type t = Stable.V1.t =
  { model : string
  ; version : string
  ; method_ : Method.t
  ; base : float
  ; margin : float
  ; score : float option
  ; top : Contribution.t list
  ; rest : float
  }
[@@deriving sexp, compare, equal]

(** Picks the [k] (default 5) largest contributions by absolute value, ties by feature
    index, skipping exact zeros. [names], [x] and [contributions] have one entry per
    feature; a NaN in [x] is a missing value. [label] maps a schema name to a
    human-readable one. *)
val create
  :  ?k:int
  -> ?label:(string -> string)
  -> ?score:float
  -> model:string
  -> version:string
  -> method_:Method.t
  -> names:string array
  -> x:float array
  -> base:float
  -> margin:float
  -> float array
  -> t

(** The stable JSON form (format 1). Field order is fixed; a missing feature value is
    [null]; floats are printed to round-trip.
    {v
    {"format":1,"model":"txn-fraud-gbdt","version":"1f0c...","method":"path_dependent",
     "base":-9.69,"margin":2.13,"score":0.81,
     "top":[{"feature":"amount","label":"amount","value":1200.0,"contribution":1.0}],
     "rest":0.12}
    v} *)
val to_yojson : t -> Yojson.Safe.t

val of_yojson : Yojson.Safe.t -> t Or_error.t
val to_json_string : t -> string

(** One line for logs and review queues, e.g.
    ["margin 2.13 (base -9.69): beneficiary_age_hours +10.39, amount +1.00, rest +0.12"]. *)
val to_string_hum : t -> string
