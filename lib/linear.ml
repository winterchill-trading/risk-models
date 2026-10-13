open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

type t =
  { intercept : float
  ; weights : float array
  ; mean : float array
  ; scale : float array
  ; log1p : bool array
  }
[@@deriving sexp_of]

let n_features t = Array.length t.weights
let intercept t = t.intercept

let create ~intercept ~weights ~mean ~scale ~log1p =
  let n = Array.length weights in
  if n = 0 || Array.length mean <> n || Array.length scale <> n || Array.length log1p <> n
  then Or_error.error_s [%message "weights, mean and scale need the same non-zero length"]
  else if not
            (List.for_all [ weights; mean; scale ] ~f:(Array.for_all ~f:Float.is_finite)
             && Float.is_finite intercept)
  then Or_error.error_s [%message "non-finite coefficient"]
  else if Array.exists scale ~f:(fun s -> Float.( <= ) s 0.)
  then Or_error.error_s [%message "scale must be positive"]
  else Ok { intercept; weights; mean; scale; log1p }
;;

let of_yojson ~n_features json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let%bind intercept = field "intercept" float json in
  let%bind weights = field "weights" (array float) json in
  let%bind mean = field "mean" (array float) json in
  let%bind scale = field "scale" (array float) json in
  let%bind log1p = field "log1p" (array int) json in
  let%bind t =
    create ~intercept ~weights ~mean ~scale ~log1p:(Array.map log1p ~f:(( <> ) 0))
  in
  if Array.length weights <> n_features
  then
    Or_error.error_s
      [%message
        "weight count differs from the schema"
          (n_features : int)
          (Array.length weights : int)]
  else Ok t
;;

(* Inlined into the loops below; out of line it would box its result. *)
let[@inline always] term t (x : float array) i =
  let v = Array.unsafe_get x i in
  if Float.is_nan v
  then 0.
  else (
    let v = if Array.unsafe_get t.log1p i then Float.log1p (Float.max v 0.) else v in
    Array.unsafe_get t.weights i
    *. ((v -. Array.unsafe_get t.mean i) /. Array.unsafe_get t.scale i))
;;

let[@zero_alloc] margin_u t x =
  if Array.length x <> Array.length t.weights
  then invalid_arg "Linear.margin: wrong feature count";
  let mutable acc = t.intercept in
  for i = 0 to Array.length t.weights - 1 do
    acc <- acc +. term t x i
  done;
  Float_u.of_float acc
;;

let[@zero_alloc] contributions_u t x ~(into : float array) =
  let n = Array.length t.weights in
  if Array.length x <> n || Array.length into <> n
  then invalid_arg "Linear.contributions: wrong feature count";
  let mutable acc = t.intercept in
  for i = 0 to n - 1 do
    let c = term t x i in
    Array.unsafe_set into i c;
    acc <- acc +. c
  done;
  Float_u.of_float acc
;;
