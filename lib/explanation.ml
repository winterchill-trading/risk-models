open! Core

module Method = struct
  type t =
    | Path_dependent
    | Interventional
    | Linear
  [@@deriving bin_io, sexp, compare, equal, enumerate]

  let to_string = function
    | Path_dependent -> "path_dependent"
    | Interventional -> "interventional"
    | Linear -> "linear"
  ;;

  let of_string s =
    match List.find all ~f:(fun m -> String.equal (to_string m) s) with
    | Some m -> Ok m
    | None -> Or_error.error_s [%message "unknown attribution method" s]
  ;;
end

module Contribution = struct
  type t =
    { feature : string
    ; label : string
    ; value : float option
    ; contribution : float
    }
  [@@deriving bin_io, sexp, compare, equal]
end

module Stable = struct
  module Method = struct
    module V1 = struct
      type t = Method.t [@@deriving bin_io, sexp, compare, equal]
    end
  end

  module Contribution = struct
    module V1 = struct
      type t = Contribution.t [@@deriving bin_io, sexp, compare, equal]
    end
  end

  module V1 = struct
    type t =
      { model : string
      ; version : string
      ; method_ : Method.V1.t
      ; base : float
      ; margin : float
      ; score : float option
      ; top : Contribution.V1.t list
      ; rest : float
      }
    [@@deriving bin_io, sexp, compare, equal]
  end
end

include Stable.V1

let create
  ?(k = 5)
  ?(label = Fn.id)
  ?score
  ~model
  ~version
  ~method_
  ~names
  ~x
  ~base
  ~margin
  contributions
  =
  let n = Array.length contributions in
  if Array.length names <> n || Array.length x <> n
  then invalid_arg "Explanation.create: names, row and contributions differ in length";
  let order =
    Array.filter (Array.init n ~f:Fn.id) ~f:(fun i -> Float.( <> ) contributions.(i) 0.)
  in
  Array.stable_sort order ~compare:(fun i j ->
    match Float.compare (Float.abs contributions.(j)) (Float.abs contributions.(i)) with
    | 0 -> Int.compare i j
    | c -> c);
  let chosen = Array.sub order ~pos:0 ~len:(Int.min k (Array.length order)) in
  let is_top = Array.create ~len:n false in
  Array.iter chosen ~f:(fun i -> is_top.(i) <- true);
  (* Summed in feature order, so [rest] does not depend on [k]'s tie-breaking. *)
  let rest = ref 0. in
  Array.iteri contributions ~f:(fun i c -> if not is_top.(i) then rest := !rest +. c);
  { model
  ; version
  ; method_
  ; base
  ; margin
  ; score
  ; top =
      Array.to_list chosen
      |> List.map ~f:(fun i ->
        { Contribution.feature = names.(i)
        ; label = label names.(i)
        ; value = Option.some_if (not (Float.is_nan x.(i))) x.(i)
        ; contribution = contributions.(i)
        })
  ; rest = !rest
  }
;;

let to_yojson t : Yojson.Safe.t =
  `Assoc
    [ "format", `Int 1
    ; "model", `String t.model
    ; "version", `String t.version
    ; "method", `String (Method.to_string t.method_)
    ; "base", `Float t.base
    ; "margin", `Float t.margin
    ; "score", Option.value_map t.score ~default:`Null ~f:(fun s -> `Float s)
    ; ( "top"
      , `List
          (List.map t.top ~f:(fun (c : Contribution.t) ->
             `Assoc
               [ "feature", `String c.feature
               ; "label", `String c.label
               ; "value", Option.value_map c.value ~default:`Null ~f:(fun v -> `Float v)
               ; "contribution", `Float c.contribution
               ])) )
    ; "rest", `Float t.rest
    ]
;;

let of_yojson json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let%bind format = field "format" int json in
  let%bind () =
    if format = 1
    then Ok ()
    else Or_error.error_s [%message "unknown explanation format" (format : int)]
  in
  let nullable_float = function
    | `Null -> Ok None
    | j -> float j >>| Option.some
  in
  let%bind model = field "model" string json in
  let%bind version = field "version" string json in
  let%bind method_ = field "method" string json >>= Method.of_string in
  let%bind base = field "base" float json in
  let%bind margin = field "margin" float json in
  let%bind score = field "score" nullable_float json in
  let%bind top =
    field
      "top"
      (list (fun j ->
         let%bind feature = field "feature" string j in
         let%bind label = field "label" string j in
         let%bind value = field "value" nullable_float j in
         let%map contribution = field "contribution" float j in
         { Contribution.feature; label; value; contribution }))
      json
  in
  let%map rest = field "rest" float json in
  { model; version; method_; base; margin; score; top; rest }
;;

let to_json_string t = Yojson.Safe.to_string (to_yojson t)

let to_string_hum t =
  let parts =
    List.map t.top ~f:(fun (c : Contribution.t) ->
      sprintf "%s %+.2f" c.label c.contribution)
    @ [ sprintf "rest %+.2f" t.rest ]
  in
  sprintf "margin %.2f (base %.2f): %s" t.margin t.base (String.concat ~sep:", " parts)
;;
