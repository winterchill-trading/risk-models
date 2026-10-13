open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

module Repr = struct
  type t =
    | Identity
    | Platt of
        { a : float
        ; b : float
        }
    | Isotonic of
        { x : float array
        ; y : float array
        }
  [@@deriving bin_io, sexp, compare, equal]
end

type t = Repr.t =
  | Identity
  | Platt of
      { a : float
      ; b : float
      }
  | Isotonic of
      { x : float array
      ; y : float array
      }
[@@deriving sexp_of, compare, equal]

let identity = Identity
let platt ~a ~b = Platt { a; b }

let isotonic ~x ~y =
  let n = Array.length x in
  let finite = Array.for_all ~f:Float.is_finite in
  if n = 0 || n <> Array.length y
  then Or_error.error_s [%message "isotonic table needs equal, non-empty x and y"]
  else if not (finite x && finite y)
  then Or_error.error_s [%message "isotonic table has a non-finite knot"]
  else if not (List.for_all (List.range 1 n) ~f:(fun i -> Float.( < ) x.(i - 1) x.(i)))
  then Or_error.error_s [%message "isotonic x must be strictly increasing"]
  else if not (List.for_all (List.range 1 n) ~f:(fun i -> Float.( <= ) y.(i - 1) y.(i)))
  then Or_error.error_s [%message "isotonic y must be non-decreasing"]
  else Ok (Isotonic { x; y })
;;

let of_yojson json =
  let open Or_error.Let_syntax in
  let open Json_util in
  match%bind field "kind" string json with
  | "identity" -> Ok Identity
  | "platt" ->
    let%bind a = field "a" float json in
    let%map b = field "b" float json in
    Platt { a; b }
  | "isotonic" ->
    let%bind x = field "x" (array float) json in
    let%bind y = field "y" (array float) json in
    isotonic ~x ~y
  | kind -> Or_error.error_s [%message "unknown calibration kind" kind]
;;

let[@zero_alloc] sigmoid_u z =
  let z = Float_u.to_float z in
  (* Branching on the sign keeps exp's argument non-positive, so it never overflows; the
     Python reference scorer does the same so the two agree to the last bit. *)
  if Float.( >= ) z 0.
  then Float_u.of_float (1. /. (1. +. Stdlib.exp (-.z)))
  else (
    let e = Stdlib.exp z in
    Float_u.of_float (e /. (1. +. e)))
;;

let sigmoid z = Float_u.to_float (sigmoid_u (Float_u.of_float z))

(* Index of the last knot <= p, for x.(0) < p < x.(n-1). *)
let[@zero_alloc] bracket (x : float array) p =
  let mutable lo = 0 in
  let mutable hi = Array.length x - 1 in
  while hi - lo > 1 do
    let mid = (lo + hi) / 2 in
    if Float.( <= ) (Array.unsafe_get x mid) p then lo <- mid else hi <- mid
  done;
  lo
;;

let[@zero_alloc] apply_u t p =
  match t with
  | Identity -> p
  | Platt { a; b } -> sigmoid_u (Float_u.of_float ((a *. Float_u.to_float p) +. b))
  | Isotonic { x; y } ->
    let p = Float_u.to_float p in
    let n = Array.length x in
    if Float.( <= ) p (Array.unsafe_get x 0)
    then Float_u.of_float (Array.unsafe_get y 0)
    else if Float.( >= ) p (Array.unsafe_get x (n - 1))
    then Float_u.of_float (Array.unsafe_get y (n - 1))
    else (
      let j = bracket x p in
      let x0 = Array.unsafe_get x j
      and x1 = Array.unsafe_get x (j + 1)
      and y0 = Array.unsafe_get y j
      and y1 = Array.unsafe_get y (j + 1) in
      let slope = (y1 -. y0) /. (x1 -. x0) in
      Float_u.of_float ((slope *. (p -. x0)) +. y0))
;;

let apply t p = Float_u.to_float (apply_u t (Float_u.of_float p))

module Stable = struct
  module V1 = struct
    type t = Repr.t [@@deriving bin_io, compare, equal, sexp_of]

    let t_of_sexp sexp =
      match Repr.t_of_sexp sexp with
      | Isotonic { x; y } -> isotonic ~x ~y |> Or_error.ok_exn
      | t -> t
    ;;
  end
end
