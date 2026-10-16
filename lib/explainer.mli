(** Per-row explanations for a loaded model: exact additive contributions plus the compact
    {!Explanation.t}.

    Trees use {!Tree_shap}: path-dependent by default (needs the node covers in the
    export; no allocation), interventional when a background sample is given. The logistic
    model's contributions are its weighted standardised terms against the training mean,
    which are already its exact additive attributions; its base is the intercept.

    An explainer owns scratch space: use one per thread or Async job. *)

open! Core

type t

(** Fails for a tree model exported without node covers (path-dependent) or a background
    of the wrong width. *)
val create : ?background:float array array -> Scorer.t -> t Or_error.t

(** For a {!Forest} outside a {!Scorer} (another library's model): [name] and [version] go
    into each explanation, [names] has one entry per feature. *)
val of_forest
  :  ?background:float array array
  -> name:string
  -> version:string
  -> names:string array
  -> Forest.t
  -> t Or_error.t

val method_ : t -> Explanation.Method.t

(** The value contributions are measured from: the forest's cover-weighted mean margin,
    the background's mean margin, or the logistic intercept. *)
val base : t -> float

(** Overwrites [into] with one contribution per feature and returns the margin;
    [base t + sum into = margin] up to rounding. Allocates nothing. *)
val contributions_u : t -> float array -> into:float array -> float#
[@@zero_alloc]

(** [k] defaults to 5. [label] gives human-readable feature names. The calibrated score is
    filled in when the explainer was made from a {!Scorer}. *)
val explain : ?k:int -> ?label:(string -> string) -> t -> float array -> Explanation.t
