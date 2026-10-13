open! Core
open Risk_models

let%expect_test "buckets, psi and status" =
  let m =
    Monitor.create ~edges:[| 0.; 0.1; 0.5; 1. |] ~reference:[| 80; 15; 5 |] ()
    |> Or_error.ok_exn
  in
  print_s [%sexp (Monitor.psi m : float option)];
  List.iter [ -1.; 0.; 0.05; 0.1; 0.49; 0.5; 1.; 2. ] ~f:(Monitor.observe m);
  print_s [%message "" ~counts:(Monitor.counts m : int array)];
  Monitor.reset m;
  (* The same mix as the reference is stable; a shift of mass to the top is not. *)
  for i = 0 to 99 do
    Monitor.observe m (if i < 80 then 0.01 else if i < 95 then 0.2 else 0.9)
  done;
  print_s
    [%message
      "" (Monitor.psi m : float option) (Monitor.status m : Monitor.Status.t option)];
  for _ = 1 to 30 do
    Monitor.observe m 0.9
  done;
  printf
    "%.4f %s\n"
    (Option.value_exn (Monitor.psi m))
    (Sexp.to_string [%sexp (Monitor.status m : Monitor.Status.t option)]);
  print_string (Monitor.to_prometheus ~labels:[ "model", "demo" ] ~name:"risk_score" m);
  [%expect
    {|
    ()
    (counts (3 2 3))
    (("Monitor.psi m" (0)) ("Monitor.status m" (Stable)))
    0.4266 (Retrain)
    # HELP risk_score Calibrated model scores.
    # TYPE risk_score histogram
    risk_score_bucket{model="demo",le="0.1"} 80
    risk_score_bucket{model="demo",le="0.5"} 95
    risk_score_bucket{model="demo",le="+Inf"} 130
    risk_score_count{model="demo"} 130
    # HELP risk_score_psi Population stability index against the calibration window.
    # TYPE risk_score_psi gauge
    risk_score_psi{model="demo"} 0.426603
    |}]
;;

let%expect_test "validation" =
  let show r =
    print_s [%sexp (Or_error.map r ~f:Monitor.observations : int Or_error.t)]
  in
  show (Monitor.create ~edges:[| 0. |] ());
  show (Monitor.create ~edges:[| 0.; 0. |] ());
  show (Monitor.create ~edges:[| 0.; 1. |] ~reference:[| 1; 2 |] ());
  [%expect
    {|
    (Error "a histogram needs at least two edges")
    (Error "edges must be strictly increasing")
    (Error "reference needs one count per bucket")
    |}]
;;

let%expect_test "the bundled model's reference against its parity fixture" =
  let model = Model.load "../models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn in
  let m = Option.value_exn (Monitor.of_model model) in
  In_channel.read_lines "fixtures/txn-fraud-gbdt.parity.csv"
  |> List.tl_exn
  |> List.iter ~f:(fun line ->
    let cols = String.split line ~on:',' |> Array.of_list in
    Monitor.observe m (Float.of_string cols.(Array.length cols - 2)));
  (* The fixture is enriched with positives (15%), so it should look like drift. *)
  print_s
    [%message
      ""
        ~observations:(Monitor.observations m : int)
        ~status:(Monitor.status m : Monitor.Status.t option)];
  [%expect {| ((observations 1000) (status (Retrain))) |}]
;;

let%expect_test "scoring into the monitor allocates nothing" =
  let model = Model.load "../models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn in
  let m = Option.value_exn (Monitor.of_model model) in
  let x = Array.create ~len:(Schema.length (Model.schema model)) 0. in
  let before = Gc.minor_words () in
  for i = 0 to 999 do
    x.(0) <- Float.of_int i;
    Monitor.observe_u m (Model.score_u model x)
  done;
  print_s [%message (Gc.minor_words () - before : int)];
  [%expect {| ("(Gc.minor_words ()) - before" 0) |}]
;;
