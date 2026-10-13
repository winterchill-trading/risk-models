open! Core
open Risk_models

(* The fixture rows come from python/export.py: features, then the Python reference
   scorer's margin, probability, calibrated score and top-3 reason indices. *)
let read_fixture name =
  let lines = In_channel.read_lines (sprintf "fixtures/%s.parity.csv" name) in
  let header = String.split (List.hd_exn lines) ~on:',' in
  let n = List.length header - 4 in
  ( List.take header n
  , List.map (List.tl_exn lines) ~f:(fun line ->
      let cols = String.split line ~on:',' |> Array.of_list in
      let f i = Float.of_string cols.(i) in
      let x = Array.init n ~f in
      let reasons =
        match cols.(n + 3) with
        | "" -> []
        | s -> String.split s ~on:';' |> List.map ~f:Int.of_string
      in
      x, f n, f (n + 1), f (n + 2), reasons) )
;;

let check name =
  let model = Model.load (sprintf "../models/%s/model.json" name) |> Or_error.ok_exn in
  let header, rows = read_fixture name in
  assert (List.equal String.equal header (Array.to_list (Model.schema model).features));
  let worst = ref 0. in
  let reason_mismatch = ref 0 in
  let nan_rows = ref 0 in
  List.iter rows ~f:(fun (x, margin, probability, score, reasons) ->
    if Array.exists x ~f:Float.is_nan then incr nan_rows;
    let s = Model.explain model x in
    List.iter
      [ s.margin -. margin
      ; Model.probability model x -. probability
      ; s.score -. score
      ; Model.score model x -. score
      ]
      ~f:(fun d -> worst := Float.max !worst (Float.abs d));
    let got =
      List.map s.reasons ~f:(fun r -> Schema.index_exn (Model.schema model) r.feature)
    in
    if not (List.equal Int.equal got reasons) then incr reason_mismatch);
  print_s
    [%message
      name
        ~rows:(List.length rows : int)
        ~rows_with_nan:(!nan_rows : int)
        ~within_1e_9:(Float.( <= ) !worst 1e-9 : bool)
        ~reason_mismatches:(!reason_mismatch : int)]
;;

let%expect_test "OCaml scores equal the Python reference within 1e-9" =
  check "txn-fraud-gbdt";
  check "txn-fraud-logistic";
  [%expect
    {|
    (txn-fraud-gbdt (rows 1000) (rows_with_nan 1000) (within_1e_9 true)
     (reason_mismatches 0))
    (txn-fraud-logistic (rows 1000) (rows_with_nan 1000) (within_1e_9 true)
     (reason_mismatches 0))
    |}]
;;
