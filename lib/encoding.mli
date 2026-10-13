(** The feature vector the models score, built from risk-rules [Features].

    risk-rules defines every feature once ([Risk_rules.Features]); this module only lays
    the record out as floats: [Features.to_floats] in [Features.numeric_names] order
    (booleans 0/1, a missing option NaN), then a one-hot encoding of the two categorical
    fields that vector leaves out, [rail], [card_channel] and [payee_check]. The Python
    training side writes vectors in the same order and records {!schema}'s digest in every
    model file. *)

open! Core

(** ["risk-rules-features"], version [Features.schema_version]. *)
val schema : Schema.t

(** The one-hot columns appended after the numeric ones. *)
val one_hot_names : string list

val encode : Risk_rules.Features.t -> float array

(** [encode] into a caller-owned array of [Schema.length schema] floats. *)
val encode_into : Risk_rules.Features.t -> float array -> unit

(** [{"name", "version", "features", "digest"}], the file the Python side reads. *)
val schema_json : unit -> Yojson.Safe.t
