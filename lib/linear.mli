(** Logistic regression on standardised features:
    [margin = intercept + sum_i weight_i * (x_i - mean_i) / scale_i], summed in feature
    order. A contribution is one term of that sum (relative to the training mean). *)

open! Core

type t [@@deriving sexp_of]

val create
  :  intercept:float
  -> weights:float array
  -> mean:float array
  -> scale:float array
  -> t Or_error.t

(** The [logistic] object of the exported model format. *)
val of_yojson : n_features:int -> Yojson.Safe.t -> t Or_error.t

val n_features : t -> int
val intercept : t -> float
val margin_u : t -> float array -> float# [@@zero_alloc]
val contributions_u : t -> float array -> into:float array -> float# [@@zero_alloc]
