open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

type t =
  { n_features : int
  ; base_score : float
  ; bias : float
  ; roots : int array
  ; feature : int array
  ; threshold : float array
  ; default_left : bool array
  ; left : int array
  ; right : int array
  ; value : float array
  }
[@@deriving sexp_of]

let n_features t = t.n_features
let n_trees t = Array.length t.roots
let n_nodes t = Array.length t.feature
let bias t = t.bias

let create
  ~n_features
  ~base_score
  ~roots
  ~feature
  ~threshold
  ~default_left
  ~left
  ~right
  ~value
  =
  let n = Array.length feature in
  let error msg i = Or_error.error_s [%message msg ~node:(i : int)] in
  let rec check i =
    if i = n
    then Ok ()
    else if feature.(i) >= n_features
    then error "feature index out of range" i
    else if feature.(i) < 0
    then check (i + 1)
    else if not (Float.is_finite threshold.(i))
    then error "non-finite threshold" i
    else if left.(i) <= i || right.(i) <= i || left.(i) >= n || right.(i) >= n
    then error "child index must follow its parent and be in range" i
    else check (i + 1)
  in
  let lengths =
    [ Array.length threshold
    ; Array.length default_left
    ; Array.length left
    ; Array.length right
    ; Array.length value
    ]
  in
  if not (List.for_all lengths ~f:(Int.equal n))
  then Or_error.error_s [%message "node arrays differ in length"]
  else if Array.is_empty roots
  then Or_error.error_s [%message "a forest needs at least one tree"]
  else if Array.exists roots ~f:(fun r -> r < 0 || r >= n)
  then Or_error.error_s [%message "root index out of range"]
  else if not (Array.for_all value ~f:Float.is_finite)
  then Or_error.error_s [%message "non-finite node value"]
  else (
    match check 0 with
    | Error _ as e -> e
    | Ok () ->
      let bias = Array.fold roots ~init:base_score ~f:(fun acc r -> acc +. value.(r)) in
      Ok
        { n_features
        ; base_score
        ; bias
        ; roots
        ; feature
        ; threshold
        ; default_left
        ; left
        ; right
        ; value
        })
;;

let of_yojson ~n_features json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let%bind base_score = field "base_score" float json in
  let%bind trees = field "trees" (list Or_error.return) json in
  let%bind parts =
    List.mapi trees ~f:(fun k tree ->
      (let%bind feature = field "feature" (array int) tree in
       let%bind threshold = field "threshold" (array float) tree in
       let%bind default_left = field "missing_left" (array int) tree in
       let%bind left = field "left" (array int) tree in
       let%bind right = field "right" (array int) tree in
       let%map value = field "value" (array float) tree in
       feature, threshold, Array.map default_left ~f:(fun b -> b <> 0), left, right, value)
      |> Or_error.tag_s ~tag:[%message "in tree" (k : int)])
    |> Or_error.combine_errors
  in
  (* Concatenate the trees, shifting child indices by each tree's offset. *)
  let offsets =
    List.folding_map parts ~init:0 ~f:(fun off (f, _, _, _, _, _) ->
      off + Array.length f, off)
  in
  let shift off a = Array.map a ~f:(fun c -> if c < 0 then c else c + off) in
  let cat f = Array.concat (List.map parts ~f) in
  create
    ~n_features
    ~base_score
    ~roots:(Array.of_list offsets)
    ~feature:(cat (fun (f, _, _, _, _, _) -> f))
    ~threshold:(cat (fun (_, t, _, _, _, _) -> t))
    ~default_left:(cat (fun (_, _, d, _, _, _) -> d))
    ~left:
      (Array.concat
         (List.map2_exn parts offsets ~f:(fun (_, _, _, l, _, _) o -> shift o l)))
    ~right:
      (Array.concat
         (List.map2_exn parts offsets ~f:(fun (_, _, _, _, r, _) o -> shift o r)))
    ~value:(cat (fun (_, _, _, _, _, v) -> v))
;;

let[@zero_alloc] [@inline] child t (x : float array) i =
  let v = Array.unsafe_get x (Array.unsafe_get t.feature i) in
  let go_left =
    if Float.is_nan v
    then Array.unsafe_get t.default_left i
    else Float.( <= ) v (Array.unsafe_get t.threshold i)
  in
  if go_left then Array.unsafe_get t.left i else Array.unsafe_get t.right i
;;

let[@zero_alloc] margin_u t (x : float array) =
  if Array.length x <> t.n_features then invalid_arg "Forest.margin: wrong feature count";
  (* Same summation order as scikit-learn: the base score, then each tree's leaf in tree
     order. Changing the order changes the last bits of the margin. *)
  let mutable acc = t.base_score in
  for k = 0 to Array.length t.roots - 1 do
    let mutable i = Array.unsafe_get t.roots k in
    while Array.unsafe_get t.feature i >= 0 do
      i <- child t x i
    done;
    acc <- acc +. Array.unsafe_get t.value i
  done;
  Float_u.of_float acc
;;

let margin t x = Float_u.to_float (margin_u t x)

let[@zero_alloc] contributions_u t (x : float array) ~(into : float array) =
  if Array.length x <> t.n_features || Array.length into <> t.n_features
  then invalid_arg "Forest.contributions: wrong feature count";
  (* A loop rather than [Array.fill]: the fill primitive is an external C call, which the
     zero_alloc checker cannot see through. *)
  for f = 0 to t.n_features - 1 do
    Array.unsafe_set into f 0.
  done;
  let mutable acc = t.base_score in
  for k = 0 to Array.length t.roots - 1 do
    let mutable i = Array.unsafe_get t.roots k in
    while Array.unsafe_get t.feature i >= 0 do
      let c = child t x i in
      let f = Array.unsafe_get t.feature i in
      Array.unsafe_set
        into
        f
        (Array.unsafe_get into f
         +. (Array.unsafe_get t.value c -. Array.unsafe_get t.value i));
      i <- c
    done;
    acc <- acc +. Array.unsafe_get t.value i
  done;
  Float_u.of_float acc
;;
