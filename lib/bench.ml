open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

type result =
  { calls : int
  ; mean_ns : float
  ; p50_ns : float
  ; p99_ns : float
  ; words_per_call : float
  }
[@@deriving sexp_of]

let percentile sorted q =
  sorted.(Int.min
            (Array.length sorted - 1)
            (Float.iround_down_exn (q *. Float.of_int (Array.length sorted))))
;;

let run rows ~calls ~f =
  if Array.is_empty rows then invalid_arg "Bench: no rows";
  let n = Array.length rows in
  for i = 0 to Int.min calls 1000 - 1 do
    f rows.(i % n)
  done;
  let words0 = Gc.minor_words () in
  let t0 = Time_ns.now () in
  for i = 0 to calls - 1 do
    f rows.(i % n)
  done;
  let elapsed = Time_ns.diff (Time_ns.now ()) t0 in
  let words = Gc.minor_words () - words0 in
  let samples = Array.create ~len:(Int.min calls 100_000) 0. in
  for i = 0 to Array.length samples - 1 do
    let a = Time_ns.now () in
    f rows.(i % n);
    samples.(i) <- Time_ns.Span.to_ns (Time_ns.diff (Time_ns.now ()) a)
  done;
  Array.sort samples ~compare:Float.compare;
  { calls
  ; mean_ns = Time_ns.Span.to_ns elapsed /. Float.of_int calls
  ; p50_ns = percentile samples 0.5
  ; p99_ns = percentile samples 0.99
  ; words_per_call = Float.of_int words /. Float.of_int calls
  }
;;

let score model rows ~calls =
  (* A float array slot stores the score unboxed; a [float ref] would box it. *)
  let sink = [| 0. |] in
  let r =
    run rows ~calls ~f:(fun x -> sink.(0) <- Float_u.to_float (Scorer.score_u model x))
  in
  ignore (Sys.opaque_identity sink : float array);
  r
;;

let explain model rows ~calls =
  run rows ~calls ~f:(fun x ->
    ignore (Sys.opaque_identity (Scorer.explain model x) : Risk_rules.Model.Score.t))
;;
