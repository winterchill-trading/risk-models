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
  let model = Scorer.load (sprintf "../models/%s/model.json" name) |> Or_error.ok_exn in
  let header, rows = read_fixture name in
  assert (List.equal String.equal header (Array.to_list (Scorer.schema model).features));
  let worst = ref 0. in
  let reason_mismatch = ref 0 in
  let nan_rows = ref 0 in
  List.iter rows ~f:(fun (x, margin, probability, score, reasons) ->
    if Array.exists x ~f:Float.is_nan then incr nan_rows;
    let s = Scorer.explain model x in
    List.iter
      [ Scorer.margin model x -. margin
      ; Scorer.probability model x -. probability
      ; s.value -. score
      ; Scorer.score model x -. score
      ]
      ~f:(fun d -> worst := Float.max !worst (Float.abs d));
    let got =
      List.map s.reasons ~f:(fun (f, _) -> Schema.index_exn (Scorer.schema model) f)
    in
    (* Reason codes may differ only where two contributions are within 1e-12 of each other
       (or of zero): libm's [log1p] differs in the last bit between macOS and Linux, which
       reorders near-ties in the logistic model's contributions. *)
    let into = Array.create ~len:(Array.length x) 0. in
    ignore (Scorer.contributions_u model x ~into : float#);
    let near a b = Float.( <= ) (Float.abs (a -. b)) 1e-12 in
    let rec agree got want =
      match got, want with
      | [], [] -> true
      | g :: gs, w :: ws -> (g = w || near into.(g) into.(w)) && agree gs ws
      | [ g ], [] | [], [ g ] -> near into.(g) 0.
      | _ -> false
    in
    if not (agree got reasons) then incr reason_mismatch);
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
  check "txn-fraud-lightgbm";
  check "txn-fraud-logistic";
  [%expect
    {|
    (txn-fraud-gbdt (rows 1000) (rows_with_nan 1000) (within_1e_9 true)
     (reason_mismatches 0))
    (txn-fraud-lightgbm (rows 1000) (rows_with_nan 1000) (within_1e_9 true)
     (reason_mismatches 0))
    (txn-fraud-logistic (rows 1000) (rows_with_nan 1000) (within_1e_9 true)
     (reason_mismatches 0))
    |}]
;;
