open! Core
open Risk_models

let iso =
  Calibration.isotonic ~x:[| 0.1; 0.2; 0.5 |] ~y:[| 0.0; 0.3; 0.9 |] |> Or_error.ok_exn
;;

let%expect_test "isotonic clips and interpolates like numpy.interp" =
  List.iter [ 0.0; 0.1; 0.15; 0.2; 0.35; 0.5; 0.7 ] ~f:(fun p ->
    printf "%.2f -> %.4f\n" p (Calibration.apply iso p));
  [%expect
    {|
    0.00 -> 0.0000
    0.10 -> 0.0000
    0.15 -> 0.1500
    0.20 -> 0.3000
    0.35 -> 0.6000
    0.50 -> 0.9000
    0.70 -> 0.9000
    |}]
;;

let%expect_test "isotonic validation" =
  let show x y =
    print_s [%sexp (Calibration.isotonic ~x ~y : Calibration.t Or_error.t)]
  in
  show [||] [||];
  show [| 0.2; 0.1 |] [| 0.; 1. |];
  show [| 0.1; 0.2 |] [| 1.; 0. |];
  show [| 0.1; Float.nan |] [| 0.; 1. |];
  [%expect
    {|
    (Error "isotonic table needs equal, non-empty x and y")
    (Error "isotonic x must be strictly increasing")
    (Error "isotonic y must be non-decreasing")
    (Error "isotonic table has a non-finite knot")
    |}]
;;

let%expect_test "sigmoid is symmetric and does not overflow" =
  List.iter [ -800.; -1.; 0.; 1.; 800. ] ~f:(fun z ->
    (* 12 digits: libm's exp may differ in the last bit between macOS and Linux. *)
    printf "%g -> %.12f\n" z (Calibration.sigmoid z));
  [%expect
    {|
    -800 -> 0.000000000000
    -1 -> 0.268941421370
    0 -> 0.500000000000
    1 -> 0.731058578630
    800 -> 1.000000000000
    |}]
;;

let%expect_test "json and stable sexp" =
  let j = Yojson.Safe.from_string {|{"kind":"platt","a":2,"b":-1.5}|} in
  let p = Calibration.of_yojson j |> Or_error.ok_exn in
  print_s [%sexp (p : Calibration.t)];
  printf "%.6f\n" (Calibration.apply p 0.75);
  print_s
    [%sexp
      (Calibration.of_yojson (`Assoc [ "kind", `String "spline" ])
       : Calibration.t Or_error.t)];
  let back = Calibration.Stable.V1.(t_of_sexp (sexp_of_t iso)) in
  print_s [%sexp (Calibration.equal iso back : bool)];
  [%expect
    {|
    (Platt (a 2) (b -1.5))
    0.500000
    (Error ("unknown calibration kind" spline))
    true
    |}]
;;

let%expect_test "apply allocates nothing" =
  let p = ref 0. in
  let before = Gc.minor_words () in
  for i = 1 to 1000 do
    p
    := !p
       +. Stdlib_upstream_compatible.Float_u.to_float
            (Calibration.apply_u
               iso
               (Stdlib_upstream_compatible.Float_u.of_float (Float.of_int i /. 1000.)))
  done;
  let words = Gc.minor_words () - before in
  print_s [%message (words : int)];
  [%expect {| (words 0) |}]
;;
