open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

(* Scratch integers written in hot loops are stored as floats. On OCaml 5 arm64 every
   store into an [int array] (and, measured, an int Bigarray) is preceded by a [dmb ishld]
   barrier; a store into a float array is a plain store. Switching an earlier node-by-node
   pass to floats took it from 45 to 31 us. Values stay below 2^53, so the conversion is
   exact. *)
module Ints = struct
  type t = float array

  let create len : t = Array.create ~len 0.
  let[@inline] [@zero_alloc] get (t : t) i = Float.to_int (Array.unsafe_get t i)
  let[@inline] [@zero_alloc] set (t : t) i v = Array.unsafe_set t i (Float.of_int v)
end

(* Path-dependent attributions, the standard way, walk every node of every tree keeping a
   "unique path" (one element per distinct feature split on so far, with the share of
   training data that followed the path through that feature's splits, whether [x]
   followed it, and the permutation weights of subsets of those features). That costs
   leaves times depth squared per tree, about 170 us for the bundled 150-tree model.

   The contribution of one leaf depends on [x] only through which of the leaf's path
   features [x] follows (all of that feature's splits on the path) and which it does not:
   a pattern of d bits for a leaf with d distinct path features. So [workspace]
   precomputes, for every leaf with d <= [max_table_depth], the contributions for all 2^d
   patterns, and a run reads each leaf's pattern off [x]'s decisions at the tree's splits
   and adds d table entries per leaf: about 30 us for that model, 9 400 table entries per
   row, which is now the bulk of the work. Leaves deeper than the cap are computed at run
   time from their precomputed path (fractions and features) by quadrature, see
   [leaf_quadrature]; a test checks both ways agree. *)

type workspace =
  { forest : Forest.t
  ; n_features : int
  ; roots : int array
  ; value : float array
  ; split_start : int array (** tree k's splits: [split_nodes.(split_start.(k)) ...] *)
  ; split_nodes : int array (** per tree, the split's index there is its decision bit *)
  ; leaf_start : int array
  ; leaf_nodes : int array
  ; leaf_d : int array (** leaf: distinct features on its path *)
  ; leaf_fo : int array (** leaf: offset of its path in [leaf_feats] and [leaf_z] *)
  ; leaf_off : int array (** leaf: offset of its table, -1 when computed at run time *)
  ; leaf_feats : int array
  ; leaf_z : float array (** share of training data following the path, per feature *)
  ; entry_mask : int array (** decision bits of the path's splits on the feature *)
  ; entry_req : int array (** the decisions (1 = left) the path takes at those splits *)
  ; table : float array
  ; gl_nodes : float array (** Gauss-Legendre rule with q points at [gl_start.(q)] *)
  ; gl_weights : float array
  ; gl_start : int array
  ; quad : float array (** scratch for {!leaf_quadrature} *)
  ; path_feature : Ints.t
  ; path_zero : float array
  ; path_one : float array
  ; path_weight : float array
  }

let forest ws = ws.forest

(* The standard path routines over the scratch path (dummy root element at index 0, the
   last element at index [depth]). *)

let[@zero_alloc] extend ws ~depth ~(zero : float#) ~(one : float#) ~feature =
  let zero = Float_u.to_float zero
  and one = Float_u.to_float one in
  let pw = ws.path_weight in
  Ints.set ws.path_feature depth feature;
  Array.unsafe_set ws.path_zero depth zero;
  Array.unsafe_set ws.path_one depth one;
  Array.unsafe_set pw depth (if depth = 0 then 1. else 0.);
  let d1 = Float.of_int (depth + 1) in
  for i = depth - 1 downto 0 do
    let w = Array.unsafe_get pw i in
    Array.unsafe_set
      pw
      (i + 1)
      (Array.unsafe_get pw (i + 1) +. (one *. w *. Float.of_int (i + 1) /. d1));
    Array.unsafe_set pw i (zero *. w *. Float.of_int (depth - i) /. d1)
  done
;;

(* The total permutation weight of the path with element [index] removed. *)
let[@zero_alloc] unwound_sum ws ~depth ~index =
  let pw = ws.path_weight in
  let one = Array.unsafe_get ws.path_one index in
  let zero = Array.unsafe_get ws.path_zero index in
  let mutable total = 0. in
  if Float.( <> ) one 0.
  then (
    let mutable next_one = Array.unsafe_get pw depth in
    for i = depth - 1 downto 0 do
      let tmp = next_one /. (Float.of_int (i + 1) *. one) in
      total <- total +. tmp;
      next_one <- Array.unsafe_get pw i -. (tmp *. zero *. Float.of_int (depth - i))
    done)
  else
    for i = depth - 1 downto 0 do
      total <- total +. (Array.unsafe_get pw i /. (zero *. Float.of_int (depth - i)))
    done;
  Float_u.of_float (total *. Float.of_int (depth + 1))
;;

(* Adds leaf [node]'s contributions for [pattern] (bit j: x follows path feature j) into
   [into], indexed by feature. *)
let[@zero_alloc] leaf_into ws ~node ~pattern (into : float array) =
  let d = Array.unsafe_get ws.leaf_d node in
  let fo = Array.unsafe_get ws.leaf_fo node in
  extend ws ~depth:0 ~zero:#1. ~one:#1. ~feature:(-1);
  for j = 0 to d - 1 do
    extend
      ws
      ~depth:(j + 1)
      ~zero:(Float_u.of_float (Array.unsafe_get ws.leaf_z (fo + j)))
      ~one:(if (pattern lsr j) land 1 = 1 then #1. else #0.)
      ~feature:(Array.unsafe_get ws.leaf_feats (fo + j))
  done;
  let v = Array.unsafe_get ws.value node in
  for i = 1 to d do
    let w = Float_u.to_float (unwound_sum ws ~depth:d ~index:i) in
    let f = Ints.get ws.path_feature i in
    Array.unsafe_set
      into
      f
      (Array.unsafe_get into f
       +. (w *. (Array.unsafe_get ws.path_one i -. Array.unsafe_get ws.path_zero i) *. v)
      )
  done
;;

(* A leaf's contribution to its j-th path feature is value * (o_j - z_j) * integral over t
   in [0, 1] of prod_[{k <> j}] (z_k (1 - t) + o_k t) (the Shapley weights |S|! (d - |S| -
   1)! / d! are Beta integrals), where z_k is the share of training data following the
   path through feature k's splits and o_k is 1 when [x] follows it. The integrand is a
   polynomial of degree d - 1, so a Gauss-Legendre rule with ceil(d / 2) points integrates
   it exactly. Used for leaves too deep to tabulate: 2.5 times faster than the path
   routines on the monitoring model's 14-feature leaves. *)
let[@zero_alloc] leaf_quadrature ws ~node ~pattern (into : float array) =
  let d = Array.unsafe_get ws.leaf_d node in
  let fo = Array.unsafe_get ws.leaf_fo node in
  let q = (d + 1) / 2 in
  let g0 = Array.unsafe_get ws.gl_start q in
  for i = 0 to q - 1 do
    let t = Array.unsafe_get ws.gl_nodes (g0 + i) in
    let mutable prod = 1. in
    for k = 0 to d - 1 do
      let z = Array.unsafe_get ws.leaf_z (fo + k) in
      let one = Float.of_int ((pattern lsr k) land 1) in
      prod <- prod *. ((z *. (1. -. t)) +. (one *. t))
    done;
    Array.unsafe_set ws.quad i prod
  done;
  let v = Array.unsafe_get ws.value node in
  for j = 0 to d - 1 do
    let z = Array.unsafe_get ws.leaf_z (fo + j) in
    let one = Float.of_int ((pattern lsr j) land 1) in
    (* The product without factor j, by division: every factor is positive for t in (0, 1)
       because covers are. Prefix and suffix products instead were slower. *)
    let mutable sum = 0. in
    for i = 0 to q - 1 do
      let t = Array.unsafe_get ws.gl_nodes (g0 + i) in
      sum
      <- sum
         +. (Array.unsafe_get ws.gl_weights (g0 + i)
             *. Array.unsafe_get ws.quad i
             /. ((z *. (1. -. t)) +. (one *. t)))
    done;
    let f = Array.unsafe_get ws.leaf_feats (fo + j) in
    Array.unsafe_set into f (Array.unsafe_get into f +. (v *. (one -. z) *. sum))
  done
;;

(* Gauss-Legendre nodes and weights on [0, 1] for [q] points: Newton's method on the
   Legendre polynomial from the usual cosine starting points. *)
let gauss_legendre q =
  let nodes = Array.create ~len:q 0.
  and weights = Array.create ~len:q 0. in
  let fq = Float.of_int q in
  for i = 0 to q - 1 do
    let x = ref (Float.cos (Float.pi *. (Float.of_int i +. 0.75) /. (fq +. 0.5))) in
    let dp = ref 0. in
    for _ = 1 to 100 do
      (* P_q(x) and its derivative by the three-term recurrence *)
      let p0 = ref 1.
      and p1 = ref !x in
      for k = 2 to q do
        let fk = Float.of_int k in
        let p2 = ((((2. *. fk) -. 1.) *. !x *. !p1) -. ((fk -. 1.) *. !p0)) /. fk in
        p0 := !p1;
        p1 := p2
      done;
      let p = if q = 1 then !x else !p1 in
      let pm1 = if q = 1 then 1. else !p0 in
      dp := fq *. ((!x *. p) -. pm1) /. ((!x *. !x) -. 1.);
      x := !x -. (p /. !dp)
    done;
    (* map [-1, 1] to [0, 1] *)
    nodes.(i) <- (1. -. !x) /. 2.;
    weights.(i) <- 1. /. ((1. -. (!x *. !x)) *. !dp *. !dp)
  done;
  nodes, weights
;;

let workspace ?(max_table_depth = 10) forest =
  match Forest.cover forest with
  | None ->
    Or_error.error_s
      [%message
        "the model has no node covers; re-export it to get path-dependent attributions"]
  | Some cover ->
    let n = Forest.n_nodes forest in
    let feature = Forest.Raw.feature forest
    and left = Forest.Raw.left forest
    and right = Forest.Raw.right forest
    and roots = Forest.Raw.roots forest in
    let leaf_d = Array.create ~len:n 0
    and leaf_fo = Array.create ~len:n 0
    and leaf_off = Array.create ~len:n (-1) in
    let feats = Queue.create ()
    and zs = Queue.create ()
    and masks = Queue.create ()
    and reqs = Queue.create ()
    and split_nodes = Queue.create ()
    and leaf_nodes = Queue.create () in
    let split_start = Array.create ~len:(Array.length roots + 1) 0
    and leaf_start = Array.create ~len:(Array.length roots + 1) 0 in
    let table_size = ref 0 in
    let too_many_splits = ref false in
    (* [path] holds, for each distinct feature above [node] in order of first appearance
       (the same for every leaf below a node): the feature, the share of training data
       that followed the path through its splits, and those splits' decision bits and
       directions. *)
    let rec build ~tree_start node (path : (int * float * int * int) list) =
      let f = feature.(node) in
      if f < 0
      then (
        Queue.enqueue leaf_nodes node;
        let path = List.rev path in
        let d = List.length path in
        leaf_d.(node) <- d;
        leaf_fo.(node) <- Queue.length feats;
        List.iter path ~f:(fun (f, z, m, r) ->
          Queue.enqueue feats f;
          Queue.enqueue zs z;
          Queue.enqueue masks m;
          Queue.enqueue reqs r);
        if d <= max_table_depth
        then (
          leaf_off.(node) <- !table_size;
          table_size := !table_size + ((1 lsl d) * d)))
      else (
        let b = Queue.length split_nodes - tree_start in
        if b > 61 then too_many_splits := true;
        let bit = 1 lsl Int.min b 61 in
        Queue.enqueue split_nodes node;
        List.iter
          [ left.(node), bit; right.(node), 0 ]
          ~f:(fun (c, req) ->
            let share = cover.(c) /. cover.(node) in
            let path =
              if List.exists path ~f:(fun (g, _, _, _) -> g = f)
              then
                List.map path ~f:(fun ((g, z, m, r) as e) ->
                  if g = f then g, z *. share, m lor bit, r lor req else e)
              else (f, share, bit, req) :: path
            in
            build ~tree_start c path))
    in
    Array.iteri roots ~f:(fun k r ->
      split_start.(k) <- Queue.length split_nodes;
      leaf_start.(k) <- Queue.length leaf_nodes;
      build ~tree_start:(Queue.length split_nodes) r []);
    split_start.(Array.length roots) <- Queue.length split_nodes;
    leaf_start.(Array.length roots) <- Queue.length leaf_nodes;
    let max_d = Array.fold leaf_d ~init:0 ~f:Int.max in
    if !too_many_splits
    then
      (* Each tree's decisions are one int of bits; 62 splits is 63 leaves. *)
      Or_error.error_s [%message "a tree has more than 62 splits"]
    else if max_d > 61
    then Or_error.error_s [%message "a leaf path has more than 61 distinct features"]
    else (
      let len = max_d + 2 in
      let max_q = (max_d + 1) / 2 in
      let rules =
        List.init (max_q + 1) ~f:(fun q -> if q = 0 then [||], [||] else gauss_legendre q)
      in
      let gl_start =
        Array.of_list
          (List.folding_map rules ~init:0 ~f:(fun acc (n, _) -> acc + Array.length n, acc))
      in
      let gl_nodes = Array.concat (List.map rules ~f:fst)
      and gl_weights = Array.concat (List.map rules ~f:snd) in
      let ws =
        { forest
        ; n_features = Forest.n_features forest
        ; roots
        ; value = Forest.Raw.value forest
        ; split_start
        ; split_nodes = Queue.to_array split_nodes
        ; leaf_start
        ; leaf_nodes = Queue.to_array leaf_nodes
        ; leaf_d
        ; leaf_fo
        ; leaf_off
        ; leaf_feats = Queue.to_array feats
        ; leaf_z = Queue.to_array zs
        ; entry_mask = Queue.to_array masks
        ; entry_req = Queue.to_array reqs
        ; table = Array.create ~len:!table_size 0.
        ; gl_nodes
        ; gl_weights
        ; gl_start
        ; quad = Array.create ~len:(max_q + 1) 0.
        ; path_feature = Ints.create len
        ; path_zero = Array.create ~len 0.
        ; path_one = Array.create ~len 0.
        ; path_weight = Array.create ~len 0.
        }
      in
      let scratch = Array.create ~len:ws.n_features 0. in
      for node = 0 to n - 1 do
        let off = leaf_off.(node) in
        if feature.(node) < 0 && off >= 0
        then (
          let d = leaf_d.(node)
          and fo = leaf_fo.(node) in
          for pattern = 0 to (1 lsl d) - 1 do
            leaf_into ws ~node ~pattern scratch;
            for j = 0 to d - 1 do
              let f = ws.leaf_feats.(fo + j) in
              ws.table.(off + (pattern * d) + j) <- scratch.(f);
              scratch.(f) <- 0.
            done
          done)
      done;
      Ok ws)
;;

let table_entries ws = Array.length ws.table

(* Per tree, one pass over its splits collects [x]'s decisions as bits of an int (1 =
   left), in registers; then each leaf reads its pattern off those bits (bit j is set when
   [x] takes the path's direction at every split on the leaf's j-th feature) and adds its
   table row. Nothing is stored between nodes, so there is no chain of dependent loads and
   stores from parent to child, and no branch on [x]'s direction. *)
let[@zero_alloc] path_dependent_u ws (x : float array) ~(into : float array) =
  let n = ws.n_features in
  if Array.length x <> n || Array.length into <> n
  then invalid_arg "Tree_shap.path_dependent: wrong feature count";
  for i = 0 to n - 1 do
    Array.unsafe_set into i 0.
  done;
  for t = 0 to Array.length ws.roots - 1 do
    let s0 = Array.unsafe_get ws.split_start t in
    let mutable decisions = 0 in
    for k = s0 to Array.unsafe_get ws.split_start (t + 1) - 1 do
      let node = Array.unsafe_get ws.split_nodes k in
      decisions
      <- decisions lor (Bool.to_int (Forest.Raw.goes_left ws.forest x node) lsl (k - s0))
    done;
    for
      k = Array.unsafe_get ws.leaf_start t to Array.unsafe_get ws.leaf_start (t + 1) - 1
    do
      let node = Array.unsafe_get ws.leaf_nodes k in
      let d = Array.unsafe_get ws.leaf_d node in
      let fo = Array.unsafe_get ws.leaf_fo node in
      let mutable p = 0 in
      for j = 0 to d - 1 do
        let m = Array.unsafe_get ws.entry_mask (fo + j) in
        let r = Array.unsafe_get ws.entry_req (fo + j) in
        p <- p lor (Bool.to_int (decisions lxor r land m = 0) lsl j)
      done;
      let off = Array.unsafe_get ws.leaf_off node in
      if off >= 0
      then (
        let base = off + (p * d) in
        for j = 0 to d - 1 do
          let g = Array.unsafe_get ws.leaf_feats (fo + j) in
          Array.unsafe_set
            into
            g
            (Array.unsafe_get into g +. Array.unsafe_get ws.table (base + j))
        done)
      else leaf_quadrature ws ~node ~pattern:p into
    done
  done;
  Forest.margin_u ws.forest x
;;

let path_dependent ws x =
  let into = Array.create ~len:ws.n_features 0. in
  let (_ : float#) = path_dependent_u ws x ~into in
  Forest.bias ws.forest, into
;;

module Background = struct
  type t =
    { forest : Forest.t
    ; rows : float array array
    ; base : float
    ; n_features : int
    ; roots : int array
    ; feature : int array
    ; value : float array
    ; width : int (** max depth + 1, the row length of [weight] *)
    ; weight : float array (** [weight.(a * width + b)] = a! b! / (a + b + 1)! *)
    ; side : Ints.t (** per feature: 0 undecided, 1 from x, 2 from the background row *)
    ; xs : Ints.t (** features taken from x on the current path *)
    ; zs : Ints.t (** features taken from the background row *)
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
        let width = Forest.max_depth forest + 1 in
        let fact k =
          List.fold
            (List.range 1 (k + 1))
            ~init:1.
            ~f:(fun acc i -> acc *. Float.of_int i)
        in
        Ok
          { forest
          ; rows = Array.map rows ~f:Array.copy
          ; base = sum /. Float.of_int (Array.length rows)
          ; n_features = n
          ; roots = Forest.Raw.roots forest
          ; feature = Forest.Raw.feature forest
          ; value = Forest.Raw.value forest
          ; width
          ; weight =
              Array.init (width * width) ~f:(fun i ->
                let a = i / width
                and b = i mod width in
                fact a *. fact b /. fact (a + b + 1))
          ; side = Ints.create n
          ; xs = Ints.create width
          ; zs = Ints.create width
          })
  ;;
end

(* Interventional: for one background row [z], walk each tree following [x] and [z]
   together. Where they part at a split on a feature not yet decided, branch: one side
   takes the feature from [x] (it joins the x-set), the other from [z]. Only one subset of
   the a + b features met that way leads exactly to a leaf, so its Shapley weights are
   closed-form: each x-set feature gets + value * (a-1)! b! / (a+b)!, each z-set feature -
   value * a! (b-1)! / (a+b)!. *)

let[@zero_alloc] rec walk (bg : Background.t) x z (into : float array) ~node ~a ~b =
  let split = Array.unsafe_get bg.feature node in
  if split < 0
  then (
    let v = Array.unsafe_get bg.value node in
    if a > 0
    then (
      let pos = v *. Array.unsafe_get bg.weight (((a - 1) * bg.width) + b) in
      for i = 0 to a - 1 do
        let f = Ints.get bg.xs i in
        Array.unsafe_set into f (Array.unsafe_get into f +. pos)
      done);
    if b > 0
    then (
      let neg = v *. Array.unsafe_get bg.weight ((a * bg.width) + b - 1) in
      for i = 0 to b - 1 do
        let f = Ints.get bg.zs i in
        Array.unsafe_set into f (Array.unsafe_get into f -. neg)
      done))
  else (
    let cx = Forest.Raw.next bg.forest x node
    and cz = Forest.Raw.next bg.forest z node in
    if cx = cz
    then walk bg x z into ~node:cx ~a ~b
    else (
      match Ints.get bg.side split with
      | 1 -> walk bg x z into ~node:cx ~a ~b
      | 2 -> walk bg x z into ~node:cz ~a ~b
      | _ ->
        Ints.set bg.side split 1;
        Ints.set bg.xs a split;
        walk bg x z into ~node:cx ~a:(a + 1) ~b;
        Ints.set bg.side split 2;
        Ints.set bg.zs b split;
        walk bg x z into ~node:cz ~a ~b:(b + 1);
        Ints.set bg.side split 0))
;;

let[@zero_alloc] interventional_u
  (bg : Background.t)
  (x : float array)
  ~(into : float array)
  =
  let n = bg.n_features in
  if Array.length x <> n || Array.length into <> n
  then invalid_arg "Tree_shap.interventional: wrong feature count";
  for i = 0 to n - 1 do
    Array.unsafe_set into i 0.
  done;
  let rows = bg.rows in
  for r = 0 to Array.length rows - 1 do
    let z = Array.unsafe_get rows r in
    for k = 0 to Array.length bg.roots - 1 do
      walk bg x z into ~node:(Array.unsafe_get bg.roots k) ~a:0 ~b:0
    done
  done;
  let k = Float.of_int (Array.length rows) in
  for i = 0 to n - 1 do
    Array.unsafe_set into i (Array.unsafe_get into i /. k)
  done;
  Forest.margin_u bg.forest x
;;

let interventional bg x ~into = Float_u.to_float (interventional_u bg x ~into)
