(** Selective classification: act on a score only when it is far enough from the decision
    threshold, abstain otherwise.

    A binary decision "positive when [score >= threshold]" is least reliable near the
    threshold. With a minimum confidence [c], scores whose confidence is below [c] are
    abstentions (in a decision system: send to a person, or fall back to rules), the rest
    are predictions. The abstentions form a band around the threshold, see {!band}.

    {!Curve} measures, on a labelled set, the accuracy (and precision, recall) of the
    predictions at every coverage (the share of rows predicted rather than abstained);
    {!for_coverage} picks the minimum confidence for a target coverage. Raising the
    confidence trades coverage for accuracy; the curve shows the price. *)

open! Core

module Confidence : sig
  type t =
    | Logit_margin
    (** distance from the threshold in log-odds, [|logit score - logit threshold|]: suited
        to calibrated probabilities spanning several orders of magnitude *)
    | Probability_margin (** [|score - threshold|] *)
  [@@deriving sexp, compare, equal, enumerate]

  (** Scores are clamped to \[1e-15, 1 - 1e-15\] before the logit. *)
  val of_score : t -> threshold:float -> float -> float
end

module Decision : sig
  type t =
    | Predict of
        { positive : bool
        ; confidence : float
        }
    | Abstain of { confidence : float }
  [@@deriving sexp, compare, equal]
end

type t =
  { threshold : float
  ; confidence : Confidence.t
  ; min_confidence : float
  }
[@@deriving sexp, compare, equal]

(** Fails unless [threshold] is in (0, 1) and [min_confidence] is finite and not negative.
    A [min_confidence] of 0 never abstains. *)
val create
  :  threshold:float
  -> confidence:Confidence.t
  -> min_confidence:float
  -> t Or_error.t

val decide : t -> float -> Decision.t

(** The scores that abstain, as \[lower, upper): empty (lower = upper = threshold) when
    [min_confidence] is 0. *)
val band : t -> float * float

module Curve : sig
  module Point : sig
    type t =
      { min_confidence : float (** predict when confidence >= this *)
      ; covered : int
      ; coverage : float (** covered / rows *)
      ; accuracy : float (** correct predictions / covered *)
      ; precision : float option (** among covered rows predicted positive *)
      ; recall : float (** covered true positives / all positives *)
      }
    [@@deriving sexp_of]
  end

  type t =
    { threshold : float
    ; confidence : Confidence.t
    ; rows : int
    ; positives : int
    ; points : Point.t array
    (** one per distinct confidence, by increasing coverage; the last covers every row *)
    }
  [@@deriving sexp_of]

  (** From [(score, label)] pairs. Fails on an empty set or a non-finite score. *)
  val compute
    :  threshold:float
    -> confidence:Confidence.t
    -> (float * bool) array
    -> t Or_error.t

  (** The first point whose coverage reaches [coverage]. *)
  val at_coverage : t -> float -> Point.t option

  (** Area under the risk-coverage curve (risk = 1 - accuracy), trapezoids over coverage:
      lower is better; a perfect ranking of mistakes last gives the least. *)
  val aurc : t -> float

  (** A table at the given coverages (default 1.0, 0.99, 0.95, 0.9, 0.8, 0.5): coverage,
      accuracy, precision, recall, and the abstain band. *)
  val to_string_hum : ?coverages:float list -> t -> string
end

(** The largest minimum confidence that still predicts at least [target] of the rows
    (target in (0, 1\]). [None] when the curve cannot reach the target. *)
val for_coverage : Curve.t -> target:float -> t option
