open! Core

type t =
  { edges : float array
  ; counts : int array
  ; reference : int array option
  ; mutable observations : int
  }

let create ?reference ~edges () =
  let n = Array.length edges in
  if n < 2
  then Or_error.error_s [%message "a histogram needs at least two edges"]
  else if not
            (List.for_all (List.range 1 n) ~f:(fun i ->
               Float.( < ) edges.(i - 1) edges.(i)))
  then Or_error.error_s [%message "edges must be strictly increasing"]
  else (
    match reference with
    | Some r when Array.length r <> n - 1 ->
      Or_error.error_s [%message "reference needs one count per bucket"]
    | _ -> Ok { edges; counts = Array.create ~len:(n - 1) 0; reference; observations = 0 })
;;

let of_model model =
  Option.map (Scorer.reference model) ~f:(fun (edges, reference) ->
    create ~reference ~edges () |> Or_error.ok_exn)
;;

let[@zero_alloc] [@inline always] bucket t s =
  let last = Array.length t.counts - 1 in
  if Float.( < ) s (Array.unsafe_get t.edges 1)
  then 0
  else if Float.( >= ) s (Array.unsafe_get t.edges last)
  then last
  else (
    (* Binary search for the bucket whose lower edge is the last one <= s. *)
    let mutable lo = 1 in
    let mutable hi = last in
    while hi - lo > 1 do
      let mid = (lo + hi) / 2 in
      if Float.( <= ) (Array.unsafe_get t.edges mid) s then lo <- mid else hi <- mid
    done;
    lo)
;;

(* Inlined into [observe_u] so the unboxed score is never boxed on the way in. *)
let[@zero_alloc] [@inline always] observe t s =
  let i = bucket t s in
  Array.unsafe_set t.counts i (Array.unsafe_get t.counts i + 1);
  t.observations <- t.observations + 1
;;

let[@zero_alloc] observe_u t s = observe t (Stdlib_upstream_compatible.Float_u.to_float s)
let observations t = t.observations
let counts t = Array.copy t.counts

let reset t =
  Array.fill t.counts ~pos:0 ~len:(Array.length t.counts) 0;
  t.observations <- 0
;;

let shares counts =
  let total = Float.of_int (Array.fold counts ~init:0 ~f:( + )) in
  Array.map counts ~f:(fun c -> Float.max 1e-4 (Float.of_int c /. total))
;;

let psi t =
  match t.reference with
  | None -> None
  | Some _ when t.observations = 0 -> None
  | Some reference ->
    let e = shares reference
    and a = shares t.counts in
    Some
      (Array.foldi a ~init:0. ~f:(fun i acc a ->
         acc +. ((a -. e.(i)) *. Float.log (a /. e.(i)))))
;;

module Status = struct
  type t =
    | Stable
    | Review
    | Retrain
  [@@deriving sexp, compare, equal]

  let of_psi p =
    if Float.( >= ) p 0.25 then Retrain else if Float.( >= ) p 0.1 then Review else Stable
  ;;
end

let status t = Option.map (psi t) ~f:Status.of_psi

let to_prometheus ?(labels = []) ~name t =
  let buf = Buffer.create 512 in
  let render extra =
    match labels @ extra with
    | [] -> ""
    | l ->
      "{"
      ^ String.concat ~sep:"," (List.map l ~f:(fun (k, v) -> sprintf "%s=%S" k v))
      ^ "}"
  in
  bprintf buf "# HELP %s Calibrated model scores.\n# TYPE %s histogram\n" name name;
  let cumulative = ref 0 in
  Array.iteri t.counts ~f:(fun i c ->
    cumulative := !cumulative + c;
    let le =
      if i = Array.length t.counts - 1 then "+Inf" else Float.to_string t.edges.(i + 1)
    in
    bprintf buf "%s_bucket%s %d\n" name (render [ "le", le ]) !cumulative);
  bprintf buf "%s_count%s %d\n" name (render []) t.observations;
  Option.iter (psi t) ~f:(fun p ->
    bprintf
      buf
      "# HELP %s_psi Population stability index against the calibration window.\n\
       # TYPE %s_psi gauge\n\
       %s_psi%s %.6f\n"
      name
      name
      name
      (render [])
      p);
  Buffer.contents buf
;;
