(** Live score distribution for drift monitoring.

    Counts calibrated scores into the same buckets as the model's reference histogram (the
    calibration window, written by [python/export.py]) and compares the two with the
    population stability index, [PSI = sum (a - e) * ln (a / e)] over bucket shares (each
    share floored at 1e-4 as on the Python side). The usual reading: below 0.1 stable,
    from 0.1 review, from 0.25 retrain. Exports Prometheus text. *)

open! Core

type t

(** Buckets are [edges.(i) <= s < edges.(i+1)], the last one closed (numpy histogram
    semantics); scores outside the edges are clamped into the first or last bucket. Fails
    on fewer than two edges, edges not strictly increasing, or a reference whose length is
    not [Array.length edges - 1]. *)
val create : ?reference:int array -> edges:float array -> unit -> t Or_error.t

(** Buckets and reference from the model file; [None] when it has no reference. *)
val of_model : Scorer.t -> t option

val observe : t -> float -> unit [@@zero_alloc]

(** [observe] taking the unboxed score [Scorer.score_u] returns; a boxed [float] argument
    costs the caller two words per call when [observe] is not inlined. *)
val observe_u : t -> float# -> unit
[@@zero_alloc]

val observations : t -> int
val counts : t -> int array
val reset : t -> unit

(** [None] without a reference or before any observation. *)
val psi : t -> float option

module Status : sig
  type t =
    | Stable (** PSI < 0.1 *)
    | Review (** 0.1 <= PSI < 0.25 *)
    | Retrain (** PSI >= 0.25 *)
  [@@deriving sexp, compare, equal]

  val of_psi : float -> t
end

val status : t -> Status.t option

(** A Prometheus histogram [<name>] (cumulative [le] buckets, [_count]; no [_sum]) and,
    with a reference, a gauge [<name>_psi]. [labels] are added to every sample. *)
val to_prometheus : ?labels:(string * string) list -> name:string -> t -> string
