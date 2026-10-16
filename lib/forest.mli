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

module Missing : sig
  (** How a split treats missing values. *)
  type t =
    | Nan_default
    (** NaN goes to the [default_left] side (scikit-learn; LightGBM [missing_type NaN]) *)
    | Nan_as_zero (** NaN is read as 0.0 and compared (LightGBM [missing_type None]) *)
    | Zero_default
    (** NaN is read as 0.0, and 0.0 (within 1e-35) goes to the [default_left] side
        (LightGBM [missing_type Zero]) *)
  [@@deriving sexp, compare, equal]
end

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
  -> ?missing:Missing.t array (** default: all [Nan_default] *)
  -> left:int array
  -> right:int array
  -> value:float array
  -> ?cover:float array
       (** training samples (or their weight) reaching each node; needed for
           path-dependent attributions ({!Tree_shap}), finite and positive *)
  -> unit
  -> t Or_error.t

(** The [gbdt] object of the exported model format, see [python/export.py]. Each tree's
    optional [count] array becomes the cover; it is used only when every tree has one. *)
val of_yojson : n_features:int -> Yojson.Safe.t -> t Or_error.t

val n_features : t -> int
val n_trees : t -> int
val n_nodes : t -> int

(** [base_score] plus the expected value of every root: the margin before any split. *)
val bias : t -> float

(** Node covers, when the export has them. *)
val cover : t -> float array option

(** The longest root-to-leaf path, in splits. *)
val max_depth : t -> int

(** Raw margin (log-odds). [x] must have [n_features t] entries (checked). *)
val margin_u : t -> float array -> float#
[@@zero_alloc]

val margin : t -> float array -> float

(** Overwrites [into] (length [n_features t]) with path attributions and returns the
    margin. *)
val contributions_u : t -> float array -> into:float array -> float#
[@@zero_alloc]

(** Read-only views of the node arrays, for attribution algorithms ({!Tree_shap}). Do not
    mutate them. *)
module Raw : sig
  val roots : t -> int array
  val feature : t -> int array
  val left : t -> int array
  val right : t -> int array
  val value : t -> float array

  (** The child of split node [i] that [x] follows (missing values per the node's
      {!Missing.t}). *)
  val next : t -> float array -> int -> int
  [@@zero_alloc]
end
