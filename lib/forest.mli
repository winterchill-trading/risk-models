(** A tree ensemble stored as flat arrays, scored without allocating.

    All trees share one set of node arrays; [roots.(k)] is tree [k]'s first node and child
    indices are absolute. A node is a leaf when [feature.(i) < 0]. At a split, a NaN input
    goes to the [default_left] side, otherwise [x <= threshold] goes left (scikit-learn
    and LightGBM semantics, compared in float64).

    [value.(i)] is the leaf value at leaves. At split nodes it is the sample-weighted
    expected value of the subtree; scoring never reads it, path attributions do: each
    split credits [value child - value node] to its feature, so [bias] plus all
    contributions equals the margin. *)

open! Core

type t [@@deriving sexp_of]

(** Validates every index (children after their parent, so walks terminate), feature index
    and threshold. *)
val create
  :  n_features:int
  -> base_score:float
  -> roots:int array
  -> feature:int array
  -> threshold:float array
  -> default_left:bool array
  -> left:int array
  -> right:int array
  -> value:float array
  -> t Or_error.t

(** The [gbdt] object of the exported model format, see [python/export.py]. *)
val of_yojson : n_features:int -> Yojson.Safe.t -> t Or_error.t

val n_features : t -> int
val n_trees : t -> int
val n_nodes : t -> int

(** [base_score] plus the expected value of every root: the margin before any split. *)
val bias : t -> float

(** Raw margin (log-odds). [x] must have [n_features t] entries (checked). *)
val margin_u : t -> float array -> float#
[@@zero_alloc]

val margin : t -> float array -> float

(** Overwrites [into] (length [n_features t]) with path attributions and returns the
    margin. *)
val contributions_u : t -> float array -> into:float array -> float#
[@@zero_alloc]
