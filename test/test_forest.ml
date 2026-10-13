open! Core
open Risk_models

(* Two trees over two features. tree 0: x0 <= 1.0 (NaN left) ? leaf -1.0 : (x1 <= 0.5 (NaN
   right) ? 0.5 : 2.0) tree 1: leaf 0.25 *)
let forest =
  Forest.create
    ~n_features:2
    ~base_score:(-3.)
    ~roots:[| 0; 5 |]
    ~feature:[| 0; -1; 1; -1; -1; -1 |]
    ~threshold:[| 1.0; 0.; 0.5; 0.; 0.; 0. |]
    ~default_left:[| true; false; false; false; false; false |]
    ~left:[| 1; -1; 3; -1; -1; -1 |]
    ~right:[| 2; -1; 4; -1; -1; -1 |]
    ~value:[| 0.1; -1.0; 1.2; 0.5; 2.0; 0.25 |]
    ()
  |> Or_error.ok_exn
;;

let%expect_test "walks, threshold equality goes left, NaN follows the default" =
  List.iter
    [ [| 0.; 0. |]
    ; [| 1.0; 9. |]
    ; [| 1.5; 0.5 |]
    ; [| 1.5; 0.6 |]
    ; [| Float.nan; 9. |]
    ; [| 2.; Float.nan |]
    ]
    ~f:(fun x ->
      let into = Array.create ~len:2 0. in
      let m = Forest.margin forest x in
      let m' =
        Stdlib_upstream_compatible.Float_u.to_float
          (Forest.contributions_u forest x ~into)
      in
      printf
        "%s -> %g  contributions %s  bias+sum %g  same %b\n"
        (Sexp.to_string [%sexp (x : float array)])
        m
        (Sexp.to_string [%sexp (into : float array)])
        (Forest.bias forest +. into.(0) +. into.(1))
        (Float.equal m m'));
  [%expect
    {|
    (0 0) -> -3.75  contributions (-1.1 0)  bias+sum -3.75  same true
    (1 9) -> -3.75  contributions (-1.1 0)  bias+sum -3.75  same true
    (1.5 0.5) -> -2.25  contributions (1.0999999999999999 -0.7)  bias+sum -2.25  same true
    (1.5 0.6) -> -0.75  contributions (1.0999999999999999 0.8)  bias+sum -0.75  same true
    (NAN 9) -> -3.75  contributions (-1.1 0)  bias+sum -3.75  same true
    (2 NAN) -> -0.75  contributions (1.0999999999999999 0.8)  bias+sum -0.75  same true
    |}]
;;

let%expect_test "validation" =
  let mk
    ?(feature = [| 0; -1; -1 |])
    ?(left = [| 1; -1; -1 |])
    ?(threshold = [| 0.; 0.; 0. |])
    ()
    =
    Forest.create
      ~n_features:1
      ~base_score:0.
      ~roots:[| 0 |]
      ~feature
      ~threshold
      ~default_left:[| true; true; true |]
      ~left
      ~right:[| 2; -1; -1 |]
      ~value:[| 0.; 1.; 2. |]
      ()
    |> Or_error.map ~f:Forest.n_nodes
  in
  let show r = print_s [%sexp (r : int Or_error.t)] in
  show (mk ());
  show (mk ~feature:[| 3; -1; -1 |] ());
  show (mk ~left:[| 0; -1; -1 |] ());
  show (mk ~threshold:[| Float.infinity; 0.; 0. |] ());
  [%expect
    {|
    (Ok 3)
    (Error ("feature index out of range" (node 0)))
    (Error ("child index must follow its parent and be in range" (node 0)))
    (Error ("non-finite threshold" (node 0)))
    |}]
;;

let%expect_test "scoring allocates nothing" =
  let x = [| 1.5; 0.6 |] in
  let into = Array.create ~len:2 0. in
  let before = Gc.minor_words () in
  for _ = 1 to 1000 do
    ignore (Sys.opaque_identity (Forest.margin_u forest x) : float#);
    ignore (Sys.opaque_identity (Forest.contributions_u forest x ~into) : float#)
  done;
  print_s [%message (Gc.minor_words () - before : int)];
  [%expect {| ("(Gc.minor_words ()) - before" 0) |}]
;;

let%expect_test "LightGBM missing modes" =
  (* One split on x0 <= -0.5, default left; leaves -1 (left) and 1 (right). *)
  let stump missing =
    Forest.create
      ~n_features:1
      ~base_score:0.
      ~roots:[| 0 |]
      ~feature:[| 0; -1; -1 |]
      ~threshold:[| -0.5; 0.; 0. |]
      ~default_left:[| true; false; false |]
      ~missing:[| missing; Nan_default; Nan_default |]
      ~left:[| 1; -1; -1 |]
      ~right:[| 2; -1; -1 |]
      ~value:[| 0.; -1.; 1. |]
      ()
    |> Or_error.ok_exn
  in
  List.iter
    Forest.Missing.[ Nan_default; Nan_as_zero; Zero_default ]
    ~f:(fun mode ->
      let f = stump mode in
      printf
        "%-12s %s\n"
        (Sexp.to_string [%sexp (mode : Forest.Missing.t)])
        (String.concat
           ~sep:" "
           (List.map [ Float.nan; 0.; -1.; 1. ] ~f:(fun v ->
              sprintf "%g->%g" v (Forest.margin f [| v |])))));
  [%expect
    {|
    Nan_default  nan->-1 0->1 -1->-1 1->1
    Nan_as_zero  nan->1 0->1 -1->-1 1->1
    Zero_default nan->-1 0->-1 -1->-1 1->1
    |}]
;;
