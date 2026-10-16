(** Exact additive per-feature attributions for a {!Forest} (SHAP values for trees).

    For a row [x], [phi.(i)] is feature [i]'s Shapley value in the game whose value of a
    feature set [S] is the forest's expected margin when the features in [S] are fixed to
    [x] and the others vary. The values add up exactly: [base + sum phi = margin x].

    Two ways to define "the others vary":

    - {b Path-dependent} ({!path_dependent_u}): the others follow the training data down
      each tree, in proportion to the node covers. [base] is {!Forest.bias}, the
      cover-weighted mean margin. This is the polynomial-time tree-path algorithm (time
      per tree proportional to leaves times depth squared), needs the covers in the model
      export, and allocates nothing.
    - {b Interventional} ({!interventional}): the others take the values of rows from a
      background sample, independently of [x]. [base] is the mean margin over the
      background. It costs one walk per tree and background row. The two differ when
      features are correlated: interventional attributions follow the model's function
      only, path-dependent ones also follow how the training data was spread along the
      tree paths.

    These are contributions to the model's log-odds, not causes: a large value says the
    model leaned on that feature for this row. *)

open! Core

(** Scratch space for one explanation at a time. Not safe to share between two concurrent
    callers. *)
type workspace

(** Fails when the forest has no node covers (an export written before covers were added). *)
val workspace : Forest.t -> workspace Or_error.t

val forest : workspace -> Forest.t

(** Overwrites [into] (length [Forest.n_features]) with the path-dependent attributions of
    [x] and returns the margin. [base] is [Forest.bias]. *)
val path_dependent_u : workspace -> float array -> into:float array -> float#
[@@zero_alloc]

(** Allocating convenience: [(base, phi)]. *)
val path_dependent : workspace -> float array -> float * float array

module Background : sig
  (** A background sample for interventional attributions. *)
  type t

  (** At least one row, each of [Forest.n_features] entries (copied). Holds scratch space:
      one explanation at a time. *)
  val create : Forest.t -> float array array -> t Or_error.t

  val forest : t -> Forest.t
  val rows : t -> int

  (** Mean margin over the rows. *)
  val base : t -> float
end

(** Overwrites [into] with the interventional attributions of [x] against the background
    (the mean over its rows) and returns the margin. Time is proportional to the number of
    background rows. *)
val interventional_u : Background.t -> float array -> into:float array -> float#
[@@zero_alloc]

val interventional : Background.t -> float array -> into:float array -> float
