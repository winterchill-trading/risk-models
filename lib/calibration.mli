(** Maps a model's raw probability to a calibrated one.

    [Isotonic] is a monotone piecewise-linear table fitted on a time window after the
    training window (scikit-learn [IsotonicRegression] with clipping): below the first
    knot it returns the first value, above the last the last value, in between it
    interpolates exactly like [numpy.interp]. [Platt] is [sigmoid (a * p + b)]. *)

open! Core

type t = private
  | Identity
  | Platt of
      { a : float
      ; b : float
      }
  | Isotonic of
      { x : float array (** strictly increasing *)
      ; y : float array (** non-decreasing, same length *)
      }
[@@deriving sexp_of, compare, equal]

val identity : t
val platt : a:float -> b:float -> t

(** Fails unless [x] is strictly increasing, [y] non-decreasing, both the same non-zero
    length and finite. *)
val isotonic : x:float array -> y:float array -> t Or_error.t

(** [{"kind": "isotonic", "x": [...], "y": [...]}], [{"kind": "platt", "a", "b"}] or
    [{"kind": "identity"}]. *)
val of_yojson : Yojson.Safe.t -> t Or_error.t

val apply : t -> float -> float

(** [apply] without boxing its result. *)
val apply_u : t -> float# -> float#
[@@zero_alloc]

(** [1 / (1 + exp (-z))], computed so neither branch overflows. *)
val sigmoid : float -> float

val sigmoid_u : float# -> float# [@@zero_alloc]

module Stable : sig
  module V1 : sig
    type nonrec t = t [@@deriving bin_io, sexp, compare, equal]
  end
end
