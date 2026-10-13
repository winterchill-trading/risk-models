(** The ordered list of feature names a model scores.

    The order is the column order of every feature vector. A model file records the
    schema it was trained on together with a digest (hex MD5 of the names joined by
    newlines, the same in Python and OCaml); loading checks the digest so a model can
    never score vectors laid out for another schema. *)

open! Core

type t = private
  { name : string
  ; version : int
  ; features : string array
  ; digest : string
  }
[@@deriving sexp_of, compare, equal]

(** Fails on an empty list or duplicate names. *)
val create : name:string -> version:int -> string list -> t Or_error.t

val create_exn : name:string -> version:int -> string list -> t

(** [digest_of_names names] is the hex MD5 of [String.concat ~sep:"\n" names]. *)
val digest_of_names : string list -> string

val length : t -> int
val index : t -> string -> int option
val index_exn : t -> string -> int

(** Fails, naming both digests, unless the two schemas list the same features in the
    same order. *)
val check_compatible : expected:t -> t -> unit Or_error.t

(** The schema the bundled models were trained on: [python/risk_models_train/schema.py]. *)
val txn_features_v1 : t

module Stable : sig
  module V1 : sig
    type nonrec t = t [@@deriving bin_io, sexp, compare, equal]
  end
end
