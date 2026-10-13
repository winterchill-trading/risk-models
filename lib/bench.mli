(** Scoring latency, for the CLI and the README numbers.

    Times [Scorer.score_u] (calibrated score, no reasons) and [Scorer.explain] (score with
    three reasons, allocating) over the given rows, round-robin. Per-call times come from
    [Time_ns.now] around each call, so they include its overhead (tens of nanoseconds);
    the mean comes from one timer around the whole loop. *)

open! Core

type result =
  { calls : int
  ; mean_ns : float (** whole loop / calls *)
  ; p50_ns : float
  ; p99_ns : float
  ; words_per_call : float (** minor-heap words *)
  }
[@@deriving sexp_of]

val score : Scorer.t -> float array array -> calls:int -> result
val explain : Scorer.t -> float array array -> calls:int -> result
