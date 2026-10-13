open! Core

let show j =
  let s = Yojson.Safe.to_string j in
  if String.length s > 40 then String.prefix s 40 ^ "..." else s
;;

let member k = function
  | `Assoc kvs ->
    (match List.Assoc.find kvs ~equal:String.equal k with
     | Some v -> Ok v
     | None -> Or_error.error_s [%message "missing field" k])
  | j -> Or_error.error_s [%message "expected an object" ~got:(show j)]
;;

let float = function
  | `Float f -> Ok f
  | `Int i -> Ok (Float.of_int i)
  | j -> Or_error.error_s [%message "expected a number" ~got:(show j)]
;;

let int = function
  | `Int i -> Ok i
  | j -> Or_error.error_s [%message "expected an integer" ~got:(show j)]
;;

let string = function
  | `String s -> Ok s
  | j -> Or_error.error_s [%message "expected a string" ~got:(show j)]
;;

let list f = function
  | `List l -> List.map l ~f |> Or_error.combine_errors
  | j -> Or_error.error_s [%message "expected a list" ~got:(show j)]
;;

let array f j = list f j |> Or_error.map ~f:Array.of_list

let field k f json =
  Or_error.bind (member k json) ~f |> Or_error.tag_s ~tag:[%message "in field" k]
;;
