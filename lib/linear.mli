(** Logistic regression on standardised features:
    [margin = intercept + sum_i weight_i * (x'_i - mean_i) / scale_i], summed in feature
    order, where [x'_i = log1p (max x_i 0)] for the features flagged [log1p] and [x_i]
    otherwise. A missing (NaN) feature was imputed with the training mean, so its term is
    zero and it is skipped. A contribution is one term of that sum. *)

open! Core

type t [@@deriving sexp_of]

val create
  :  intercept:float
  -> weights:float array
  -> mean:float array
  -> scale:float array
  -> log1p:bool array
  -> t Or_error.t

(** The [logistic] object of the exported model format. *)
val of_yojson : n_features:int -> Yojson.Safe.t -> t Or_error.t

val n_features : t -> int
val intercept : t -> float
val margin_u : t -> float array -> float# [@@zero_alloc]
val contributions_u : t -> float array -> into:float array -> float# [@@zero_alloc]
