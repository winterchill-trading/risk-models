(** Small readers over [Yojson.Safe.t] that return [Or_error.t] with the field name. *)

open! Core

val member : string -> Yojson.Safe.t -> Yojson.Safe.t Or_error.t
val float : Yojson.Safe.t -> float Or_error.t
val int : Yojson.Safe.t -> int Or_error.t
val string : Yojson.Safe.t -> string Or_error.t
val list : (Yojson.Safe.t -> 'a Or_error.t) -> Yojson.Safe.t -> 'a list Or_error.t
val array : (Yojson.Safe.t -> 'a Or_error.t) -> Yojson.Safe.t -> 'a array Or_error.t

(** [field k f json] reads member [k] with [f], tagging errors with [k]. *)
val field : string -> (Yojson.Safe.t -> 'a Or_error.t) -> Yojson.Safe.t -> 'a Or_error.t
