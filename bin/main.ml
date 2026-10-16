open! Core
open Risk_models

let model_arg = Command.Param.(anon ("MODEL" %: Filename_unix.arg_type))
let csv_arg = Command.Param.(anon ("CSV" %: Filename_unix.arg_type))

let load path =
  match Scorer.load path with
  | Ok m -> m
  | Error e ->
    eprint_s [%sexp (e : Error.t)];
    exit 1
;;

(* Rows from a CSV whose header names features of the model's schema, in any order;
   missing columns and empty cells are NaN. Extra columns (labels, expected scores) are
   ignored. *)
let read_rows model path =
  let schema = Scorer.schema model in
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

let encode_cmd =
  Command.basic
    ~summary:
      "Read risk-rules Features JSON lines on stdin, print the model vectors as CSV"
    ~readme:(fun () ->
      "Each input line is one Risk_rules.Features.to_json record (what `risk-rules \
       features` prints for a context). Output: a header with the encoding's feature \
       names, then one row per record; missing values are empty cells. This is the path \
       from journaled contexts to training rows, so training and serving share \
       Features.of_context.")
    (Command.Param.return (fun () ->
       print_endline (String.concat ~sep:"," (Array.to_list Encoding.schema.features));
       In_channel.iter_lines In_channel.stdin ~f:(fun line ->
         if not (String.is_empty (String.strip line))
         then (
           match Risk_rules.Features.of_json line with
           | Error e ->
             eprint_s [%sexp (e : Error.t)];
             exit 1
           | Ok f ->
             Encoding.encode f
             |> Array.to_list
             |> List.map ~f:(fun v -> if Float.is_nan v then "" else sprintf "%.17g" v)
             |> String.concat ~sep:","
             |> print_endline))))
;;

let check_cmd =
  Command.basic
    ~summary:"Load a model, check its schema against risk-rules Features, summarise it"
    (let%map_open.Command path = model_arg in
     fun () ->
       let m = load path in
       let kind =
         match Scorer.body m with
         | Forest f ->
           sprintf "gbdt, %d trees, %d nodes" (Forest.n_trees f) (Forest.n_nodes f)
         | Linear _ -> "logistic"
       in
       print_s
         [%message
           (Scorer.name m)
             ~version:(Scorer.version m : string)
             (kind : string)
             ~schema:((Scorer.schema m).digest : string)
             ~thresholds:(Scorer.thresholds m : Scorer.Thresholds.t option)])
;;

let score_cmd =
  Command.basic
    ~summary:"Score every row of a CSV of features; prints score and top reasons"
    (let%map_open.Command path = model_arg
     and csv = csv_arg in
     fun () ->
       let m = load path in
       Array.iter (read_rows m csv) ~f:(fun x ->
         let s = Scorer.explain m x in
         printf
           "%.6f\t%s\n"
           s.value
           (String.concat
              ~sep:" "
              (List.map s.reasons ~f:(fun (f, c) -> sprintf "%s:%.3f" f c)))))
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
       let ok = Or_error.ok_exn in
       let path_dependent = Explainer.create m |> Or_error.ok in
       (* Interventional cost grows with the background, so it runs fewer calls. *)
       let background = Array.sub rows ~pos:0 ~len:(Int.min 100 (Array.length rows)) in
       let interventional =
         match Scorer.body m with
         | Forest _ -> Some (Explainer.create ~background m |> ok)
         | Linear _ -> None
       in
       print_s
         [%message
           (Scorer.name m)
             ~score:(Bench.score m rows ~calls : Bench.result)
             ~explain:(Bench.explain m rows ~calls:(calls / 10) : Bench.result)
             ~attribute:
               (Option.map path_dependent ~f:(fun e ->
                  ( Explanation.Method.to_string (Explainer.method_ e)
                  , Bench.attribute e rows ~calls:(calls / 10) ))
                : (string * Bench.result) option)
             ~explanation:
               (Option.map path_dependent ~f:(fun e ->
                  Bench.explanation e rows ~calls:(calls / 10))
                : Bench.result option)
             ~interventional:
               (Option.map interventional ~f:(fun e ->
                  Bench.attribute e rows ~calls:(Int.max 1 (calls / 1000)))
                : Bench.result option)
             ~background_rows:(Array.length background : int)])
;;

(* "amount=1200,mcc=5999": features named in the model's schema; the rest are missing. *)
let parse_features m spec =
  let schema = Scorer.schema m in
  let x = Array.create ~len:(Schema.length schema) Float.nan in
  String.split spec ~on:','
  |> List.filter ~f:(fun s -> not (String.is_empty (String.strip s)))
  |> List.iter ~f:(fun kv ->
    match String.lsplit2 kv ~on:'=' with
    | None -> failwithf "expected name=value, got %S" kv ()
    | Some (name, v) ->
      let name = String.strip name in
      (match Schema.index schema name with
       | None -> failwithf "unknown feature %S (see risk-models schema)" name ()
       | Some i ->
         x.(i)
         <- (match String.strip v with
             | "" | "nan" | "NaN" -> Float.nan
             | v -> Float.of_string v)));
  x
;;

let explain_cmd =
  Command.basic
    ~summary:"Explain a score: base value, exact per-feature contributions, top k"
    ~readme:(fun () ->
      "Prints one Explanation JSON line (format 1) per row: the base value, the \
       margin,        the calibrated score, the k features with the largest \
       contributions to the        log-odds and the sum of the rest; base + \
       contributions = margin. Trees use        path-dependent attributions (the \
       training covers in the export), or        interventional ones against -background \
       rows. These are model contributions, not        causes.")
    (let%map_open.Command path =
       flag "-model" (required Filename_unix.arg_type) ~doc:"FILE model.json"
     and features =
       flag
         "-features"
         (optional string)
         ~doc:"NAME=VALUE,... one row; unnamed features are missing"
     and csv =
       flag "-csv" (optional Filename_unix.arg_type) ~doc:"FILE rows to explain (CSV)"
     and background =
       flag
         "-background"
         (optional Filename_unix.arg_type)
         ~doc:"FILE background rows (CSV) for interventional attributions"
     and k = flag "-k" (optional_with_default 5 int) ~doc:"N features to list (5)" in
     fun () ->
       let m = load path in
       let rows =
         Option.to_list (Option.map features ~f:(parse_features m))
         @ Option.value_map csv ~default:[] ~f:(fun f -> Array.to_list (read_rows m f))
       in
       if List.is_empty rows then failwith "give -features or -csv";
       let explainer =
         Explainer.create ?background:(Option.map background ~f:(read_rows m)) m
         |> Or_error.ok_exn
       in
       List.iter rows ~f:(fun x ->
         print_endline (Explanation.to_json_string (Explainer.explain ~k explainer x))))
;;

let () =
  Command_unix.run
    (Command.group
       ~summary:"Score payments with risk-models exports"
       [ "schema", schema_cmd
       ; "encode", encode_cmd
       ; "check", check_cmd
       ; "score", score_cmd
       ; "explain", explain_cmd
       ; "bench", bench_cmd
       ])
;;
