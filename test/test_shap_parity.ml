open! Core
open Risk_models

(* [fixtures/<model>.shap.csv] comes from python/export.py: the shap library's
   TreeExplainer on the same model, path-dependent for the first 200 parity rows and
   interventional for 50 rows against a background of 100 others, whose indices are the
   "background" lines. *)

let rows name =
  let lines = In_channel.read_lines (sprintf "fixtures/%s.parity.csv" name) in
  let n = (String.split (List.hd_exn lines) ~on:',' |> List.length) - 4 in
  List.tl_exn lines
  |> List.map ~f:(fun line ->
    let cols = String.split line ~on:',' |> Array.of_list in
    Array.init n ~f:(fun i -> Float.of_string cols.(i)))
  |> Array.of_list
;;

let shap_rows name =
  In_channel.read_lines (sprintf "fixtures/%s.shap.csv" name)
  |> List.tl_exn
  |> List.map ~f:(fun line ->
    match String.split line ~on:',' with
    | "background" :: row :: _ -> `Background (Int.of_string row)
    | meth :: row :: base :: phi ->
      `Explained
        ( meth
        , Int.of_string row
        , Float.of_string base
        , Array.of_list_map phi ~f:Float.of_string )
    | _ -> failwith "bad shap fixture line")
;;

let forest name =
  let model = Scorer.load (sprintf "../models/%s/model.json" name) |> Or_error.ok_exn in
  match Scorer.body model with
  | Forest f -> f
  | Linear _ -> failwith "not a tree model"
;;

let compare_model name =
  let f = forest name in
  let x = rows name in
  let ws = Tree_shap.workspace f |> Or_error.ok_exn in
  let lines = shap_rows name in
  let background =
    List.filter_map lines ~f:(function
      | `Background i -> Some x.(i)
      | `Explained _ -> None)
    |> Array.of_list
    |> Tree_shap.Background.create f
    |> Or_error.ok_exn
  in
  let worst = String.Table.create () in
  let note meth d =
    Hashtbl.update worst meth ~f:(fun w -> Float.max d (Option.value w ~default:0.))
  in
  let counts = String.Table.create () in
  List.iter lines ~f:(function
    | `Background _ -> ()
    | `Explained (meth, row, base, expected) ->
      Hashtbl.incr counts meth;
      let ours_base, phi =
        match meth with
        | "path" -> Tree_shap.path_dependent ws x.(row)
        | "interventional" ->
          let into = Array.create ~len:(Forest.n_features f) 0. in
          let (_ : float) = Tree_shap.interventional background x.(row) ~into in
          Tree_shap.Background.base background, into
        | m -> failwithf "unknown method %s" m ()
      in
      note meth (Float.abs (ours_base -. base));
      Array.iter2_exn phi expected ~f:(fun a b -> note meth (Float.abs (a -. b))));
  (* Path-dependent agrees to rounding (3.6e-15 measured on macOS arm64). shap's
     interventional mode adds in single precision (a leaf of 1 + 1e-12 comes back as 1),
     so there the bound is shap's own error: 4.6e-7 measured, 1e-6 required. *)
  Hashtbl.to_alist worst
  |> List.sort ~compare:[%compare: string * float]
  |> List.iter ~f:(fun (meth, d) ->
    let bound = if String.equal meth "path" then 1e-12 else 1e-6 in
    printf
      "%s %s: %d rows, max |diff| <= %g: %b\n"
      name
      meth
      (Hashtbl.find_exn counts meth)
      bound
      Float.(d <= bound))
;;

let%expect_test "path-dependent and interventional attributions match shap" =
  compare_model "txn-fraud-gbdt";
  compare_model "txn-fraud-lightgbm";
  [%expect
    {|
    txn-fraud-gbdt interventional: 50 rows, max |diff| <= 1e-06: true
    txn-fraud-gbdt path: 200 rows, max |diff| <= 1e-12: true
    txn-fraud-lightgbm interventional: 50 rows, max |diff| <= 1e-06: true
    txn-fraud-lightgbm path: 200 rows, max |diff| <= 1e-12: true
    |}]
;;
