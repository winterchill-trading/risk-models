open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

(* The path-dependent algorithm keeps, for the current root-to-node path, the "unique
   path": one element per distinct feature split on so far (plus a dummy for the root),
   each with the fraction of zero paths (the share of training data that followed this
   path through that feature's splits) and one paths (1 when [x] followed it, else 0), and
   the permutation weights of subsets of those features. A recursive call copies its
   parent's path into the next segment of the arrays below, so no allocation happens while
   walking. *)

type workspace =
  { forest : Forest.t
  ; cover : float array
  ; n_features : int
  ; roots : int array
  ; feature : int array
  ; left : int array
  ; right : int array
  ; value : float array
  ; path_feature : int array
  ; path_zero : float array
  ; path_one : float array
  ; path_weight : float array
  }

let forest ws = ws.forest

let workspace forest =
  match Forest.cover forest with
  | None ->
    Or_error.error_s
      [%message
        "the model has no node covers; re-export it to get path-dependent attributions"]
  | Some cover ->
    (* A call at depth [d] uses a segment of at most [d + 2] elements starting after its
       parent's, so the segments of one root-to-leaf path fit in (D+2)(D+3)/2. *)
    let d = Forest.max_depth forest in
    let len = ((d + 2) * (d + 3) / 2) + 1 in
    Ok
      { forest
      ; cover
      ; n_features = Forest.n_features forest
      ; roots = Forest.Raw.roots forest
      ; feature = Forest.Raw.feature forest
      ; left = Forest.Raw.left forest
      ; right = Forest.Raw.right forest
      ; value = Forest.Raw.value forest
      ; path_feature = Array.create ~len (-1)
      ; path_zero = Array.create ~len 0.
      ; path_one = Array.create ~len 0.
      ; path_weight = Array.create ~len 0.
      }
;;

(* The three routines below are the standard ones (extend, unwind, unwound sum), over a
   segment starting at [s] whose last element has index [s + depth]. *)

let[@zero_alloc] extend ws ~s ~depth ~zero ~one ~feature =
  let pw = ws.path_weight in
  Array.unsafe_set ws.path_feature (s + depth) feature;
  Array.unsafe_set ws.path_zero (s + depth) (Float_u.to_float zero);
  Array.unsafe_set ws.path_one (s + depth) (Float_u.to_float one);
  Array.unsafe_set pw (s + depth) (if depth = 0 then 1. else 0.);
  let zero = Float_u.to_float zero
  and one = Float_u.to_float one in
  let d1 = Float.of_int (depth + 1) in
  for i = depth - 1 downto 0 do
    let w = Array.unsafe_get pw (s + i) in
    Array.unsafe_set
      pw
      (s + i + 1)
      (Array.unsafe_get pw (s + i + 1) +. (one *. w *. Float.of_int (i + 1) /. d1));
    Array.unsafe_set pw (s + i) (zero *. w *. Float.of_int (depth - i) /. d1)
  done
;;

let[@zero_alloc] unwind ws ~s ~depth ~index =
  let pw = ws.path_weight in
  let one = Array.unsafe_get ws.path_one (s + index) in
  let zero = Array.unsafe_get ws.path_zero (s + index) in
  let d1 = Float.of_int (depth + 1) in
  let mutable next_one = Array.unsafe_get pw (s + depth) in
  for i = depth - 1 downto 0 do
    if Float.( <> ) one 0.
    then (
      let w = Array.unsafe_get pw (s + i) in
      let nw = next_one *. d1 /. (Float.of_int (i + 1) *. one) in
      Array.unsafe_set pw (s + i) nw;
      next_one <- w -. (nw *. zero *. Float.of_int (depth - i) /. d1))
    else
      Array.unsafe_set
        pw
        (s + i)
        (Array.unsafe_get pw (s + i) *. d1 /. (zero *. Float.of_int (depth - i)))
  done;
  for i = index to depth - 1 do
    Array.unsafe_set
      ws.path_feature
      (s + i)
      (Array.unsafe_get ws.path_feature (s + i + 1));
    Array.unsafe_set ws.path_zero (s + i) (Array.unsafe_get ws.path_zero (s + i + 1));
    Array.unsafe_set ws.path_one (s + i) (Array.unsafe_get ws.path_one (s + i + 1))
  done
;;

(* The total permutation weight of the path with element [index] removed, without
   modifying the path. *)
let[@zero_alloc] unwound_sum ws ~s ~depth ~index =
  let pw = ws.path_weight in
  let one = Array.unsafe_get ws.path_one (s + index) in
  let zero = Array.unsafe_get ws.path_zero (s + index) in
  let mutable total = 0. in
  if Float.( <> ) one 0.
  then (
    let mutable next_one = Array.unsafe_get pw (s + depth) in
    for i = depth - 1 downto 0 do
      let tmp = next_one /. (Float.of_int (i + 1) *. one) in
      total <- total +. tmp;
      next_one <- Array.unsafe_get pw (s + i) -. (tmp *. zero *. Float.of_int (depth - i))
    done)
  else
    for i = depth - 1 downto 0 do
      total <- total +. (Array.unsafe_get pw (s + i) /. (zero *. Float.of_int (depth - i)))
    done;
  Float_u.of_float (total *. Float.of_int (depth + 1))
;;

(* [parent] is the start of the parent's segment; this call's segment starts right after
   the parent's [depth] elements (the parent had [depth - 1] plus the dummy). *)
let[@zero_alloc] rec recurse
  ws
  (x : float array)
  (into : float array)
  ~node
  ~parent
  ~depth
  ~(zero : float#)
  ~(one : float#)
  ~feature
  =
  let f = ws.forest in
  let s = parent + depth + 1 in
  for i = 0 to depth do
    Array.unsafe_set
      ws.path_feature
      (s + i)
      (Array.unsafe_get ws.path_feature (parent + i));
    Array.unsafe_set ws.path_zero (s + i) (Array.unsafe_get ws.path_zero (parent + i));
    Array.unsafe_set ws.path_one (s + i) (Array.unsafe_get ws.path_one (parent + i));
    Array.unsafe_set ws.path_weight (s + i) (Array.unsafe_get ws.path_weight (parent + i))
  done;
  extend ws ~s ~depth ~zero ~one ~feature;
  let split = Array.unsafe_get ws.feature node in
  if split < 0
  then (
    let leaf = Array.unsafe_get ws.value node in
    for i = 1 to depth do
      let w = Float_u.to_float (unwound_sum ws ~s ~depth ~index:i) in
      let fi = Array.unsafe_get ws.path_feature (s + i) in
      Array.unsafe_set
        into
        fi
        (Array.unsafe_get into fi
         +. (w
             *. (Array.unsafe_get ws.path_one (s + i)
                 -. Array.unsafe_get ws.path_zero (s + i))
             *. leaf))
    done)
  else (
    let hot = Forest.Raw.next f x node in
    let left = Array.unsafe_get ws.left node in
    let cold = if hot = left then Array.unsafe_get ws.right node else left in
    let c = Array.unsafe_get ws.cover node in
    let hot_zero = Array.unsafe_get ws.cover hot /. c in
    let cold_zero = Array.unsafe_get ws.cover cold /. c in
    (* A feature split on again further down: undo its earlier element and redo it here
       with the product of both fractions. *)
    let mutable index = 0 in
    while index <= depth && Array.unsafe_get ws.path_feature (s + index) <> split do
      index <- index + 1
    done;
    let mutable incoming_zero = 1. in
    let mutable incoming_one = 1. in
    let mutable depth = depth in
    if index <= depth
    then (
      incoming_zero <- Array.unsafe_get ws.path_zero (s + index);
      incoming_one <- Array.unsafe_get ws.path_one (s + index);
      unwind ws ~s ~depth ~index;
      depth <- depth - 1);
    recurse
      ws
      x
      into
      ~node:hot
      ~parent:s
      ~depth:(depth + 1)
      ~zero:(Float_u.of_float (hot_zero *. incoming_zero))
      ~one:(Float_u.of_float incoming_one)
      ~feature:split;
    recurse
      ws
      x
      into
      ~node:cold
      ~parent:s
      ~depth:(depth + 1)
      ~zero:(Float_u.of_float (cold_zero *. incoming_zero))
      ~one:#0.
      ~feature:split)
;;

let[@zero_alloc] path_dependent_u ws (x : float array) ~(into : float array) =
  let n = ws.n_features in
  if Array.length x <> n || Array.length into <> n
  then invalid_arg "Tree_shap.path_dependent: wrong feature count";
  for i = 0 to n - 1 do
    Array.unsafe_set into i 0.
  done;
  let roots = ws.roots in
  for k = 0 to Array.length roots - 1 do
    (* The root extends a path of length zero with a dummy element (feature -1); its
       segment starts at 1, and the index-0 "parent" element it copies is overwritten. *)
    recurse
      ws
      x
      into
      ~node:(Array.unsafe_get roots k)
      ~parent:0
      ~depth:0
      ~zero:#1.
      ~one:#1.
      ~feature:(-1)
  done;
  Forest.margin_u ws.forest x
;;

let path_dependent ws x =
  let into = Array.create ~len:(Forest.n_features ws.forest) 0. in
  let (_ : float#) = path_dependent_u ws x ~into in
  Forest.bias ws.forest, into
;;

module Background = struct
  type t =
    { forest : Forest.t
    ; rows : float array array
    ; base : float
    }

  let forest t = t.forest
  let rows t = Array.length t.rows
  let base t = t.base

  let create forest rows =
    let n = Forest.n_features forest in
    if Array.is_empty rows
    then Or_error.error_s [%message "the background sample is empty"]
    else (
      match Array.findi rows ~f:(fun _ r -> Array.length r <> n) with
      | Some (i, r) ->
        Or_error.error_s
          [%message
            "background row has the wrong length"
              (i : int)
              ~length:(Array.length r : int)
              ~expected:(n : int)]
      | None ->
        let sum = Array.sum (module Float) rows ~f:(Forest.margin forest) in
        Ok { forest; rows; base = sum /. Float.of_int (Array.length rows) })
  ;;
end

(* Interventional: for one background row [z], walk each tree following [x] and [z]
   together. Where they part at a split on a feature not yet decided, branch: one side
   takes the feature from [x] (it joins the x-set), the other from [z]. Only one subset of
   the a + b features met that way leads exactly to a leaf, so its Shapley weights are
   closed-form: each x-set feature gets + value * (a-1)! b! / (a+b)!, each z-set feature -
   value * a! (b-1)! / (a+b)!. *)

let interventional (bg : Background.t) (x : float array) ~(into : float array) =
  let f = bg.forest in
  let n = Forest.n_features f in
  if Array.length x <> n || Array.length into <> n
  then invalid_arg "Tree_shap.interventional: wrong feature count";
  Array.fill into ~pos:0 ~len:n 0.;
  let d = Forest.max_depth f in
  (* weight.(a).(b) = a! b! / (a + b + 1)! *)
  let weight =
    Array.init (d + 1) ~f:(fun a ->
      Array.init (d + 1) ~f:(fun b ->
        let fact k =
          List.fold
            (List.range 1 (k + 1))
            ~init:1.
            ~f:(fun acc i -> acc *. Float.of_int i)
        in
        fact a *. fact b /. fact (a + b + 1)))
  in
  (* 0 undecided, 1 from x, 2 from z *)
  let side = Array.create ~len:n 0 in
  let xs = Array.create ~len:(d + 1) 0
  and zs = Array.create ~len:(d + 1) 0 in
  let feature = Forest.Raw.feature f
  and value = Forest.Raw.value f in
  let rec walk z node ~a ~b =
    let split = feature.(node) in
    if split < 0
    then (
      let v = value.(node) in
      if a > 0
      then (
        let pos = v *. weight.(a - 1).(b) in
        for i = 0 to a - 1 do
          into.(xs.(i)) <- into.(xs.(i)) +. pos
        done);
      if b > 0
      then (
        let neg = v *. weight.(a).(b - 1) in
        for i = 0 to b - 1 do
          into.(zs.(i)) <- into.(zs.(i)) -. neg
        done))
    else (
      let cx = Forest.Raw.next f x node
      and cz = Forest.Raw.next f z node in
      if cx = cz
      then walk z cx ~a ~b
      else (
        match side.(split) with
        | 1 -> walk z cx ~a ~b
        | 2 -> walk z cz ~a ~b
        | _ ->
          side.(split) <- 1;
          xs.(a) <- split;
          walk z cx ~a:(a + 1) ~b;
          side.(split) <- 2;
          zs.(b) <- split;
          walk z cz ~a ~b:(b + 1);
          side.(split) <- 0))
  in
  let roots = Forest.Raw.roots f in
  Array.iter bg.rows ~f:(fun z -> Array.iter roots ~f:(fun r -> walk z r ~a:0 ~b:0));
  let k = Float.of_int (Array.length bg.rows) in
  for i = 0 to n - 1 do
    into.(i) <- into.(i) /. k
  done;
  Forest.margin f x
;;
