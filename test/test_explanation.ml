open! Core
open Risk_models

let example =
  Explanation.create
    ~k:2
    ~label:(function
      | "amount" -> "payment amount"
      | s -> s)
    ~score:0.25
    ~model:"m"
    ~version:"v1"
    ~method_:Path_dependent
    ~names:[| "amount"; "age"; "country"; "unused" |]
    ~x:[| 1200.; Float.nan; 1.; 0. |]
    ~base:(-3.)
    ~margin:(-0.5)
    [| 1.5; -2.; 1.0; 0. |]
;;

let%expect_test "top k by absolute contribution, the rest summed" =
  print_s [%sexp (example : Explanation.t)];
  print_endline (Explanation.to_string_hum example);
  [%expect
    {|
    ((model m) (version v1) (method_ Path_dependent) (base -3) (margin -0.5)
     (score (0.25))
     (top
      (((feature age) (label age) (value ()) (contribution -2))
       ((feature amount) (label "payment amount") (value (1200))
        (contribution 1.5))))
     (rest 1))
    margin -0.50 (base -3.00): age -2.00, payment amount +1.50, rest +1.00
    |}]
;;

let%expect_test "stable JSON round-trips" =
  let json = Explanation.to_json_string example in
  print_endline json;
  let back = Explanation.of_yojson (Yojson.Safe.from_string json) |> Or_error.ok_exn in
  printf "equal after round trip: %b\n" (Explanation.equal example back);
  [%expect
    {|
    {"format":1,"model":"m","version":"v1","method":"path_dependent","base":-3.0,"margin":-0.5,"score":0.25,"top":[{"feature":"age","label":"age","value":null,"contribution":-2.0},{"feature":"amount","label":"payment amount","value":1200.0,"contribution":1.5}],"rest":1.0}
    equal after round trip: true
    |}]
;;

let%expect_test "Quickcheck: base + top + rest equals the margin" =
  let gen =
    let open Quickcheck.Generator.Let_syntax in
    let%bind n = Int.gen_incl 1 12 in
    let%bind k = Int.gen_incl 0 15 in
    let%map c = List.gen_with_length n (Float.gen_incl (-5.) 5.) in
    k, Array.of_list c
  in
  Quickcheck.test gen ~sexp_of:[%sexp_of: int * float array] ~f:(fun (k, c) ->
    let n = Array.length c in
    let margin = 0.5 +. Array.sum (module Float) c ~f:Fn.id in
    let e =
      Explanation.create
        ~k
        ~model:"m"
        ~version:"v"
        ~method_:Linear
        ~names:(Array.init n ~f:Int.to_string)
        ~x:(Array.create ~len:n 0.)
        ~base:0.5
        ~margin
        c
    in
    let total =
      e.base +. e.rest +. List.sum (module Float) e.top ~f:(fun t -> t.contribution)
    in
    assert (List.length e.top <= k);
    assert (Float.(abs (total -. margin) < 1e-9)))
;;

let rows name =
  let lines = In_channel.read_lines (sprintf "fixtures/%s.parity.csv" name) in
  let n = (String.split (List.hd_exn lines) ~on:',' |> List.length) - 4 in
  List.tl_exn lines
  |> List.map ~f:(fun line ->
    let cols = String.split line ~on:',' |> Array.of_list in
    Array.init n ~f:(fun i -> Float.of_string cols.(i)))
;;

let%expect_test "explainers add up on every fixture row, for every model" =
  List.iter
    [ "txn-fraud-gbdt"; "txn-fraud-lightgbm"; "txn-fraud-logistic" ]
    ~f:(fun name ->
      let scorer =
        Scorer.load (sprintf "../models/%s/model.json" name) |> Or_error.ok_exn
      in
      let xs = rows name in
      let explainer = Explainer.create scorer |> Or_error.ok_exn in
      let worst = ref 0. in
      List.iter xs ~f:(fun x ->
        let e = Explainer.explain ~k:3 explainer x in
        let total =
          e.base +. e.rest +. List.sum (module Float) e.top ~f:(fun t -> t.contribution)
        in
        worst := Float.max !worst (Float.abs (total -. Scorer.margin scorer x));
        assert (Float.equal (Option.value_exn e.score) (Scorer.score scorer x)));
      printf
        "%s %s: %d rows, |base + contributions - margin| < 1e-9: %b\n"
        name
        (Explanation.Method.to_string (Explainer.method_ explainer))
        (List.length xs)
        Float.(!worst < 1e-9));
  [%expect
    {|
    txn-fraud-gbdt path_dependent: 1000 rows, |base + contributions - margin| < 1e-9: true
    txn-fraud-lightgbm path_dependent: 1000 rows, |base + contributions - margin| < 1e-9: true
    txn-fraud-logistic linear: 1000 rows, |base + contributions - margin| < 1e-9: true
    |}]
;;

let%expect_test "an export without node covers explains only interventionally" =
  let json = Yojson.Safe.from_file "../models/txn-fraud-gbdt/model.json" in
  let rec drop_counts : Yojson.Safe.t -> Yojson.Safe.t = function
    | `Assoc kv ->
      `Assoc
        (List.filter_map kv ~f:(fun (k, v) ->
           if String.equal k "count" then None else Some (k, drop_counts v)))
    | `List l -> `List (List.map l ~f:drop_counts)
    | j -> j
  in
  let json = drop_counts json in
  let scorer = Scorer.of_yojson json |> Or_error.ok_exn in
  let x = List.hd_exn (rows "txn-fraud-gbdt") in
  print_s
    [%sexp
      (Explainer.create scorer |> Or_error.map ~f:Explainer.method_
       : Explanation.Method.t Or_error.t)];
  let e = Explainer.create ~background:[| x |] scorer |> Or_error.ok_exn in
  print_s [%sexp (Explainer.method_ e : Explanation.Method.t)];
  [%expect
    {|
    (Error
     "the model has no node covers; re-export it to get path-dependent attributions")
    Interventional
    |}]
;;

let%expect_test "explaining allocates nothing" =
  let scorer = Scorer.load "../models/txn-fraud-gbdt/model.json" |> Or_error.ok_exn in
  let xs = rows "txn-fraud-gbdt" in
  let x = List.hd_exn xs in
  let path = Explainer.create scorer |> Or_error.ok_exn in
  let interventional =
    Explainer.create ~background:(Array.of_list (List.take xs 10)) scorer
    |> Or_error.ok_exn
  in
  let into = Array.create ~len:(Array.length x) 0. in
  List.iter
    [ "path", path; "interventional", interventional ]
    ~f:(fun (name, e) ->
      let (_ : float#) = Explainer.contributions_u e x ~into in
      let before = Gc.minor_words () in
      for _ = 1 to 10 do
        let (_ : float#) = Explainer.contributions_u e x ~into in
        ()
      done;
      printf "%s: %d words\n" name (Gc.minor_words () - before));
  [%expect {|
    path: 0 words
    interventional: 0 words
    |}]
;;
