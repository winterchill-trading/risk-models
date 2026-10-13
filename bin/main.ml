open! Core
open Risk_models

let model_arg = Command.Param.(anon ("MODEL" %: Filename_unix.arg_type))
let csv_arg = Command.Param.(anon ("CSV" %: Filename_unix.arg_type))

let load path =
  match Model.load path with
  | Ok m -> m
  | Error e ->
    eprint_s [%sexp (e : Error.t)];
    exit 1
;;

(* Rows from a CSV whose header names features of the model's schema, in any order;
   missing columns and empty cells are NaN. Extra columns (labels, expected scores) are
   ignored. *)
let read_rows model path =
  let schema = Model.schema model in
  match In_channel.read_lines path with
  | [] -> [||]
  | header :: lines ->
    let cols =
      String.split header ~on:',' |> List.map ~f:(Schema.index schema) |> Array.of_list
    in
    List.map lines ~f:(fun line ->
      let x = Array.create ~len:(Schema.length schema) Float.nan in
      List.iteri (String.split line ~on:',') ~f:(fun j cell ->
        match cols.(j) with
        | Some i when not (String.is_empty cell) -> x.(i) <- Float.of_string cell
        | _ -> ());
      x)
    |> Array.of_list
;;

let schema_cmd =
  Command.basic
    ~summary:"Print the feature schema the models score, as JSON (read by python/)"
    (Command.Param.return (fun () ->
       print_endline (Yojson.Safe.pretty_to_string (Encoding.schema_json ()))))
;;

let check_cmd =
  Command.basic
    ~summary:"Load a model, check its schema against risk-rules Features, summarise it"
    (let%map_open.Command path = model_arg in
     fun () ->
       let m = load path in
       let kind =
         match Model.body m with
         | Forest f ->
           sprintf "gbdt, %d trees, %d nodes" (Forest.n_trees f) (Forest.n_nodes f)
         | Linear _ -> "logistic"
       in
       print_s
         [%message
           (Model.name m)
             ~version:(Model.version m : string)
             (kind : string)
             ~schema:((Model.schema m).digest : string)
             ~thresholds:(Model.thresholds m : Model.Thresholds.t option)])
;;

let score_cmd =
  Command.basic
    ~summary:"Score every row of a CSV of features; prints score and top reasons"
    (let%map_open.Command path = model_arg
     and csv = csv_arg in
     fun () ->
       let m = load path in
       Array.iter (read_rows m csv) ~f:(fun x ->
         let s = Model.explain m x in
         printf
           "%.6f\t%s\n"
           s.score
           (String.concat
              ~sep:" "
              (List.map s.reasons ~f:(fun r -> sprintf "%s:%.3f" r.feature r.contribution)))))
;;

let bench_cmd =
  Command.basic
    ~summary:"Measure scoring latency over the rows of a CSV of features"
    (let%map_open.Command path = model_arg
     and csv = csv_arg
     and calls =
       flag
         "-calls"
         (optional_with_default 1_000_000 int)
         ~doc:"N calls (default 1000000)"
     in
     fun () ->
       let m = load path in
       let rows = read_rows m csv in
       print_s
         [%message
           (Model.name m)
             ~score:(Bench.score m rows ~calls : Bench.result)
             ~explain:(Bench.explain m rows ~calls:(calls / 10) : Bench.result)])
;;

let () =
  Command_unix.run
    (Command.group
       ~summary:"Score payments with risk-models exports"
       [ "schema", schema_cmd
       ; "check", check_cmd
       ; "score", score_cmd
       ; "bench", bench_cmd
       ])
;;
