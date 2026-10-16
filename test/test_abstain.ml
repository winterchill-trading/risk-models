open! Core
open Risk_models

let%expect_test "decide and band" =
  let t =
    Abstain.create ~threshold:0.5 ~confidence:Probability_margin ~min_confidence:0.1
    |> Or_error.ok_exn
  in
  List.iter [ 0.05; 0.45; 0.5; 0.55; 0.65; 0.95 ] ~f:(fun s ->
    match Abstain.decide t s with
    | Predict { positive; confidence } ->
      printf "%.2f predict %b (confidence %.3f)\n" s positive confidence
    | Abstain { confidence } -> printf "%.2f abstain (confidence %.3f)\n" s confidence);
  print_s [%sexp (Abstain.band t : float * float)];
  let l =
    Abstain.create ~threshold:0.001 ~confidence:Logit_margin ~min_confidence:1.
    |> Or_error.ok_exn
  in
  let lo, hi = Abstain.band l in
  printf "logit band around 0.001: [%.6f, %.6f)\n" lo hi;
  print_s
    [%sexp
      (Abstain.create ~threshold:1. ~confidence:Logit_margin ~min_confidence:1.
       : Abstain.t Or_error.t)];
  [%expect
    {|
    0.05 predict false (confidence 0.450)
    0.45 abstain (confidence 0.050)
    0.50 abstain (confidence 0.000)
    0.55 abstain (confidence 0.050)
    0.65 predict true (confidence 0.150)
    0.95 predict true (confidence 0.450)
    (0.4 0.6)
    logit band around 0.001: [0.000368, 0.002714)
    (Error ("threshold must be in (0, 1)" (threshold 1)))
    |}]
;;

(* Labels drawn so that rows far from the threshold are right more often: the curve must
   rise as coverage falls. *)
let synthetic n =
  let rand = Random.State.make [| 5 |] in
  Array.init n ~f:(fun _ ->
    let s = Random.State.float rand 1. in
    let p_positive = s in
    s, Float.(Random.State.float rand 1. < p_positive))
;;

let%expect_test "accuracy-coverage curve and the threshold for a target coverage" =
  let data = synthetic 10_000 in
  let curve =
    Abstain.Curve.compute ~threshold:0.5 ~confidence:Probability_margin data
    |> Or_error.ok_exn
  in
  print_string (Abstain.Curve.to_string_hum curve);
  printf "aurc %.4f\n" (Abstain.Curve.aurc curve);
  let t = Abstain.for_coverage curve ~target:0.8 |> Option.value_exn in
  let predicted =
    Array.count data ~f:(fun (s, _) ->
      match Abstain.decide t s with
      | Predict _ -> true
      | Abstain _ -> false)
  in
  printf
    "for 80%% coverage: min_confidence %.4f, predicts %d of %d rows\n"
    t.min_confidence
    predicted
    (Array.length data);
  [%expect
    {|
    accuracy-coverage at threshold 0.5 (probability margin confidence): 10000 rows, 4969 positive
      coverage 1.000: accuracy 0.7449, precision 0.744, recall 0.743, abstain on [0.5, 0.5)
      coverage 0.990: accuracy 0.7468, precision 0.746, recall 0.736, abstain on [0.495, 0.505)
      coverage 0.950: accuracy 0.7563, precision 0.754, recall 0.713, abstain on [0.475, 0.525)
      coverage 0.900: accuracy 0.7679, precision 0.767, recall 0.685, abstain on [0.45, 0.55)
      coverage 0.800: accuracy 0.7933, precision 0.790, recall 0.634, abstain on [0.401, 0.599)
      coverage 0.500: accuracy 0.8796, precision 0.877, recall 0.438, abstain on [0.251, 0.749)
    aurc 0.1234
    for 80% coverage: min_confidence 0.0992, predicts 8000 of 10000 rows
    |}]
;;

let%expect_test "Quickcheck: curve invariants" =
  let gen =
    let open Quickcheck.Generator.Let_syntax in
    let%bind n = Int.gen_incl 1 60 in
    let%map rows =
      List.gen_with_length
        n
        (Quickcheck.Generator.tuple2 (Float.gen_incl 0. 1.) Bool.quickcheck_generator)
    in
    Array.of_list rows
  in
  Quickcheck.test gen ~sexp_of:[%sexp_of: (float * bool) array] ~f:(fun data ->
    List.iter Abstain.Confidence.all ~f:(fun confidence ->
      let curve =
        Abstain.Curve.compute ~threshold:0.3 ~confidence data |> Or_error.ok_exn
      in
      let points = curve.points in
      let last = Array.last_exn points in
      (* full coverage is plain accuracy *)
      let plain =
        Float.of_int (Array.count data ~f:(fun (s, y) -> Bool.equal Float.(s >= 0.3) y))
        /. Float.of_int (Array.length data)
      in
      assert (last.covered = Array.length data);
      assert (Float.(abs (last.accuracy -. plain) < 1e-12));
      (* coverage strictly increases, confidence strictly decreases *)
      Array.iteri points ~f:(fun i p ->
        if i > 0
        then (
          assert (p.covered > points.(i - 1).covered);
          let previous = points.(i - 1) in
          assert (Float.(p.min_confidence < previous.min_confidence))));
      (* the chosen cut-off predicts exactly the covered rows of its point *)
      List.iter [ 0.1; 0.5; 0.9; 1.0 ] ~f:(fun target ->
        let t = Abstain.for_coverage curve ~target |> Option.value_exn in
        let predicted =
          Array.count data ~f:(fun (s, _) ->
            match Abstain.decide t s with
            | Predict _ -> true
            | Abstain _ -> false)
        in
        assert (
          Float.(
            Float.of_int predicted /. Float.of_int (Array.length data) >= target -. 1e-12)))))
;;
