open! Core

module Confidence = struct
  type t =
    | Logit_margin
    | Probability_margin
  [@@deriving sexp, compare, equal, enumerate]

  let logit p =
    let p = Float.clamp_exn p ~min:1e-15 ~max:(1. -. 1e-15) in
    Float.log (p /. (1. -. p))
  ;;

  let of_score t ~threshold score =
    match t with
    | Logit_margin -> Float.abs (logit score -. logit threshold)
    | Probability_margin -> Float.abs (score -. threshold)
  ;;
end

module Decision = struct
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

let create ~threshold ~confidence ~min_confidence =
  if not Float.(threshold > 0. && threshold < 1.)
  then Or_error.error_s [%message "threshold must be in (0, 1)" (threshold : float)]
  else if not Float.(is_finite min_confidence && min_confidence >= 0.)
  then
    Or_error.error_s
      [%message "min_confidence must be finite and not negative" (min_confidence : float)]
  else Ok { threshold; confidence; min_confidence }
;;

let decide t score : Decision.t =
  let confidence = Confidence.of_score t.confidence ~threshold:t.threshold score in
  if Float.(confidence >= t.min_confidence)
  then Predict { positive = Float.(score >= t.threshold); confidence }
  else Abstain { confidence }
;;

let band t =
  if Float.(t.min_confidence = 0.)
  then t.threshold, t.threshold
  else (
    match t.confidence with
    | Probability_margin ->
      ( Float.max 0. (t.threshold -. t.min_confidence)
      , Float.min 1. (t.threshold +. t.min_confidence) )
    | Logit_margin ->
      let sigmoid z = 1. /. (1. +. Float.exp (-.z)) in
      let l = Confidence.logit t.threshold in
      sigmoid (l -. t.min_confidence), sigmoid (l +. t.min_confidence))
;;

module Curve = struct
  module Point = struct
    type t =
      { min_confidence : float
      ; covered : int
      ; coverage : float
      ; accuracy : float
      ; precision : float option
      ; recall : float
      }
    [@@deriving sexp_of]
  end

  type t =
    { threshold : float
    ; confidence : Confidence.t
    ; rows : int
    ; positives : int
    ; points : Point.t array
    }
  [@@deriving sexp_of]

  let compute ~threshold ~confidence (data : (float * bool) array) =
    if Array.is_empty data
    then Or_error.error_s [%message "no labelled rows"]
    else if Array.exists data ~f:(fun (s, _) -> not (Float.is_finite s))
    then Or_error.error_s [%message "non-finite score"]
    else (
      let rows = Array.length data in
      let positives = Array.count data ~f:snd in
      let scored =
        Array.map data ~f:(fun (s, y) ->
          Confidence.of_score confidence ~threshold s, Float.(s >= threshold), y)
      in
      (* Most confident first; a point closes every group of equal confidence, since a
         cut-off cannot separate them. *)
      Array.sort scored ~compare:(fun (a, _, _) (b, _, _) -> Float.compare b a);
      let points = Queue.create () in
      let correct = ref 0
      and predicted_positive = ref 0
      and true_positive = ref 0 in
      Array.iteri scored ~f:(fun i (c, predicted, y) ->
        if Bool.equal predicted y then incr correct;
        if predicted then incr predicted_positive;
        if predicted && y then incr true_positive;
        let last_of_group =
          i = rows - 1
          ||
          let c', _, _ = scored.(i + 1) in
          Float.( <> ) c c'
        in
        if last_of_group
        then (
          let covered = i + 1 in
          Queue.enqueue
            points
            { Point.min_confidence = c
            ; covered
            ; coverage = Float.of_int covered /. Float.of_int rows
            ; accuracy = Float.of_int !correct /. Float.of_int covered
            ; precision =
                (if !predicted_positive = 0
                 then None
                 else
                   Some (Float.of_int !true_positive /. Float.of_int !predicted_positive))
            ; recall =
                (if positives = 0
                 then 0.
                 else Float.of_int !true_positive /. Float.of_int positives)
            }));
      Ok { threshold; confidence; rows; positives; points = Queue.to_array points })
  ;;

  let at_coverage t coverage =
    Array.find t.points ~f:(fun p -> Float.(p.coverage >= coverage -. 1e-12))
  ;;

  let aurc t =
    (* Starts from the first point (the curve is undefined at zero coverage). *)
    let risk (p : Point.t) = 1. -. p.accuracy in
    let area = ref 0. in
    for i = 1 to Array.length t.points - 1 do
      let a = t.points.(i - 1)
      and b = t.points.(i) in
      area := !area +. ((b.coverage -. a.coverage) *. (risk a +. risk b) /. 2.)
    done;
    !area
  ;;

  let to_string_hum ?(coverages = [ 1.0; 0.99; 0.95; 0.9; 0.8; 0.5 ]) t =
    let header =
      sprintf
        "accuracy-coverage at threshold %g (%s confidence): %d rows, %d positive"
        t.threshold
        (match t.confidence with
         | Logit_margin -> "logit margin"
         | Probability_margin -> "probability margin")
        t.rows
        t.positives
    in
    let line target =
      match at_coverage t target with
      | None -> sprintf "  coverage %.2f: not reached" target
      | Some p ->
        let lo, hi =
          band
            { threshold = t.threshold
            ; confidence = t.confidence
            ; min_confidence = p.min_confidence
            }
        in
        sprintf
          "  coverage %.3f: accuracy %.4f, precision %s, recall %.3f, abstain on [%.3g, \
           %.3g)"
          p.coverage
          p.accuracy
          (Option.value_map p.precision ~default:"-" ~f:(sprintf "%.3f"))
          p.recall
          lo
          hi
    in
    String.concat ~sep:"\n" ((header :: List.map coverages ~f:line) @ [ "" ])
  ;;
end

let for_coverage (curve : Curve.t) ~target =
  if not Float.(target > 0. && target <= 1.)
  then None
  else
    Curve.at_coverage curve target
    |> Option.map ~f:(fun (p : Curve.Point.t) ->
      { threshold = curve.threshold
      ; confidence = curve.confidence
      ; min_confidence = p.min_confidence
      })
;;
