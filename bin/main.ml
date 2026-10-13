open! Core
open Risk_models

let schema_cmd =
  Command.basic
    ~summary:"Print the feature schema the models score, as JSON (read by python/)"
    (Command.Param.return (fun () ->
       print_endline (Yojson.Safe.pretty_to_string (Encoding.schema_json ()))))
;;

let () =
  Command_unix.run
    (Command.group
       ~summary:"Score payments with risk-models exports"
       [ "schema", schema_cmd ])
;;
