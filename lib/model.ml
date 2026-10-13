open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

module Body = struct
  type t =
    | Forest of Forest.t
    | Linear of Linear.t
  [@@deriving sexp_of]
end

module Thresholds = struct
  type t =
    { review : float
    ; block : float option
    ; alerts_per_day : int
    ; block_precision : float
    }
  [@@deriving sexp, compare, equal]
end

type t =
  { name : string
  ; schema : Schema.t
  ; body : Body.t
  ; calibration : Calibration.t
  ; thresholds : Thresholds.t option
  ; reference : (float array * int array) option
  }
[@@deriving sexp_of]

let name t = t.name
let schema t = t.schema
let body t = t.body
let calibration t = t.calibration
let thresholds t = t.thresholds
let reference t = t.reference

let create ?thresholds ?reference ~name ~schema body calibration =
  let n = Schema.length schema in
  let body_n =
    match (body : Body.t) with
    | Forest f -> Forest.n_features f
    | Linear l -> Linear.n_features l
  in
  if body_n <> n
  then
    Or_error.error_s
      [%message "model and schema feature counts differ" (n : int) (body_n : int)]
  else (
    match reference with
    | Some (edges, counts) when Array.length edges <> Array.length counts + 1 ->
      Or_error.error_s [%message "reference histogram needs one more edge than counts"]
    | _ -> Ok { name; schema; body; calibration; thresholds; reference })
;;

let schema_of_yojson json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let%bind name = field "name" string json in
  let%bind version = field "version" int json in
  let%bind features = field "features" (list string) json in
  let%bind digest = field "digest" string json in
  let%bind schema = Schema.create ~name ~version features in
  if String.equal digest schema.digest
  then Ok schema
  else
    Or_error.error_s
      [%message "schema digest does not match its feature names" digest schema.digest]
;;

let of_yojson ?(expected = Encoding.schema) json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let%bind format = field "format" int json in
  let%bind () =
    if format = 1
    then Ok ()
    else Or_error.error_s [%message "unsupported model format" (format : int)]
  in
  let%bind name = field "name" string json in
  let%bind schema = field "schema" schema_of_yojson json in
  let%bind () = Schema.check_compatible ~expected schema in
  let n_features = Schema.length schema in
  let%bind link = field "link" string json in
  let%bind () =
    if String.equal link "logistic"
    then Ok ()
    else Or_error.error_s [%message "unsupported link" link]
  in
  let%bind body =
    match%bind field "kind" string json with
    | "gbdt" ->
      field "gbdt" (Forest.of_yojson ~n_features) json >>| fun f -> Body.Forest f
    | "logistic" ->
      field "logistic" (Linear.of_yojson ~n_features) json >>| fun l -> Body.Linear l
    | kind -> Or_error.error_s [%message "unknown model kind" kind]
  in
  let%bind calibration = field "calibration" Calibration.of_yojson json in
  let optional k f =
    match member k json with
    | Error _ | Ok `Null -> Ok None
    | Ok j -> f j >>| Option.some |> Or_error.tag_s ~tag:[%message "in field" k]
  in
  let%bind reference =
    optional "reference" (fun j ->
      let%bind edges = field "edges" (array float) j in
      let%map counts = field "counts" (array int) j in
      edges, counts)
  in
  let%bind thresholds =
    optional "thresholds" (fun j ->
      let%bind review = field "review" float j in
      let%bind block =
        field
          "block"
          (function
            | `Null -> Ok None
            | j -> float j >>| Option.some)
          j
      in
      let%bind alerts_per_day = field "alerts_per_day" int j in
      let%map block_precision = field "block_precision" float j in
      { Thresholds.review; block; alerts_per_day; block_precision })
  in
  create ?thresholds ?reference ~name ~schema body calibration
;;

let load ?expected path =
  Or_error.try_with_join (fun () -> of_yojson ?expected (Yojson.Safe.from_file path))
  |> Or_error.tag_s ~tag:[%message "loading model" path]
;;

let[@zero_alloc] margin_u t x =
  match t.body with
  | Forest f -> Forest.margin_u f x
  | Linear l -> Linear.margin_u l x
;;

let[@zero_alloc] probability_u t x = Calibration.sigmoid_u (margin_u t x)
let[@zero_alloc] score_u t x = Calibration.apply_u t.calibration (probability_u t x)
let margin t x = Float_u.to_float (margin_u t x)
let probability t x = Float_u.to_float (probability_u t x)
let score t x = Float_u.to_float (score_u t x)

let[@zero_alloc] contributions_u t x ~into =
  match t.body with
  | Forest f -> Forest.contributions_u f x ~into
  | Linear l -> Linear.contributions_u l x ~into
;;

let explain ?(k = 3) t x =
  let into = Array.create ~len:(Schema.length t.schema) 0. in
  let margin = Float_u.to_float (contributions_u t x ~into) in
  let probability = Calibration.sigmoid margin in
  { Score.model = t.name
  ; score = Calibration.apply t.calibration probability
  ; margin
  ; reasons = Score.top_reasons ~names:t.schema.features into ~k
  }
;;
