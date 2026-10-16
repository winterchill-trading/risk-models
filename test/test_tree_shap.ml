open! Core
open Risk_models
module Float_u = Stdlib_upstream_compatible.Float_u

(* Random forests small enough to check against the Shapley formula by enumerating every
   feature subset. Covers add up (parent = left + right), as in a trained export, and
   values at split nodes are the cover-weighted means of their children. *)

let random_forest ~n_features ~trees ~max_depth rand =
  let feature = Queue.create ()
  and threshold = Queue.create ()
  and default_left = Queue.create ()
  and left = Queue.create ()
  and right = Queue.create ()
  and value = Queue.create ()
  and cover = Queue.create () in
  (* returns (index, value, cover) *)
  let rec node depth =
    let i = Queue.length feature in
    List.iter [ feature; left; right ] ~f:(fun q -> Queue.enqueue q (-1));
    Queue.enqueue threshold 0.;
    Queue.enqueue default_left (Random.State.bool rand);
    Queue.enqueue value 0.;
    Queue.enqueue cover 0.;
    if depth >= max_depth || (depth > 0 && Random.State.int rand 4 = 0)
    then (
      let v = Random.State.float_range rand (-1.) 1.
      and c = Float.of_int (1 + Random.State.int rand 20) in
      Queue.set value i v;
      Queue.set cover i c;
      i, v, c)
    else (
      Queue.set feature i (Random.State.int rand n_features);
      (* thresholds on a small grid, so rows land on both sides and on the threshold *)
      Queue.set threshold i (Float.of_int (Random.State.int rand 4) /. 2.);
      let l, vl, cl = node (depth + 1) in
      let r, vr, cr = node (depth + 1) in
      Queue.set left i l;
      Queue.set right i r;
      let c = cl +. cr in
      let v = ((cl *. vl) +. (cr *. vr)) /. c in
      Queue.set value i v;
      Queue.set cover i c;
      i, v, c)
  in
  let roots = Array.init trees ~f:(fun _ -> Tuple3.get1 (node 0)) in
  Forest.create
    ~n_features
    ~base_score:(Random.State.float_range rand (-2.) 0.)
    ~roots
    ~feature:(Queue.to_array feature)
    ~threshold:(Queue.to_array threshold)
    ~default_left:(Queue.to_array default_left)
    ~left:(Queue.to_array left)
    ~right:(Queue.to_array right)
    ~value:(Queue.to_array value)
    ~cover:(Queue.to_array cover)
    ()
  |> Or_error.ok_exn
;;

let random_row ~n_features rand =
  Array.init n_features ~f:(fun _ ->
    match Random.State.int rand 6 with
    | 0 -> Float.nan
    | k -> Float.of_int (k - 1) /. 2.)
;;

(* Shapley values of the game [v] over [n] players, by enumerating subsets. *)
let shapley ~n v =
  let fact k =
    List.fold (List.range 1 (k + 1)) ~init:1. ~f:(fun a i -> a *. Float.of_int i)
  in
  let phi = Array.create ~len:n 0. in
  for mask = 0 to (1 lsl n) - 1 do
    let size = Int.popcount mask in
    let vs = v mask in
    for i = 0 to n - 1 do
      if mask land (1 lsl i) = 0
      then (
        let w = fact size *. fact (n - size - 1) /. fact n in
        phi.(i) <- phi.(i) +. (w *. (v (mask lor (1 lsl i)) -. vs)))
    done
  done;
  phi
;;

(* The path-dependent game: features in the set follow [x], the others split the walk in
   proportion to the children's covers. *)
let path_game forest x mask =
  let feature = Forest.Raw.feature forest
  and left = Forest.Raw.left forest
  and right = Forest.Raw.right forest
  and value = Forest.Raw.value forest
  and cover = Forest.cover forest |> Option.value_exn in
  let rec g i =
    let f = feature.(i) in
    if f < 0
    then value.(i)
    else if mask land (1 lsl f) <> 0
    then g (Forest.Raw.next forest x i)
    else (
      let l = left.(i)
      and r = right.(i) in
      ((cover.(l) *. g l) +. (cover.(r) *. g r)) /. cover.(i))
  in
  Array.fold (Forest.Raw.roots forest) ~init:0. ~f:(fun acc r -> acc +. g r)
;;

(* The interventional game: features in the set from [x], the others from each background
   row, averaged. *)
let interventional_game forest ~background x mask =
  let n = Forest.n_features forest in
  Array.sum (module Float) background ~f:(fun z ->
    Forest.margin
      forest
      (Array.init n ~f:(fun i -> if mask land (1 lsl i) <> 0 then x.(i) else z.(i))))
  /. Float.of_int (Array.length background)
;;

let worst_diff a b =
  Array.fold2_exn a b ~init:0. ~f:(fun acc a b -> Float.max acc (Float.abs (a -. b)))
;;

let%expect_test "path-dependent attributions equal enumerated Shapley values" =
  let rand = Random.State.make [| 42 |] in
  let worst = ref 0.
  and worst_sum = ref 0. in
  for _ = 1 to 200 do
    let n_features = 1 + Random.State.int rand 6 in
    let forest =
      random_forest
        ~n_features
        ~trees:(1 + Random.State.int rand 3)
        ~max_depth:(1 + Random.State.int rand 6)
        rand
    in
    let ws = Tree_shap.workspace forest |> Or_error.ok_exn in
    for _ = 1 to 5 do
      let x = random_row ~n_features rand in
      let base, phi = Tree_shap.path_dependent ws x in
      let expected = shapley ~n:n_features (path_game forest x) in
      worst := Float.max !worst (worst_diff phi expected);
      let sum = base +. Array.sum (module Float) phi ~f:Fn.id in
      worst_sum := Float.max !worst_sum (Float.abs (sum -. Forest.margin forest x))
    done
  done;
  printf "max |phi - shapley| < 1e-12: %b\n" Float.(!worst < 1e-12);
  printf "max |base + sum phi - margin| < 1e-12: %b\n" Float.(!worst_sum < 1e-12);
  [%expect
    {|
    max |phi - shapley| < 1e-12: true
    max |base + sum phi - margin| < 1e-12: true
    |}]
;;

let%expect_test "interventional attributions equal enumerated Shapley values" =
  let rand = Random.State.make [| 7 |] in
  let worst = ref 0.
  and worst_sum = ref 0. in
  for _ = 1 to 100 do
    let n_features = 1 + Random.State.int rand 6 in
    let forest =
      random_forest
        ~n_features
        ~trees:(1 + Random.State.int rand 3)
        ~max_depth:(1 + Random.State.int rand 6)
        rand
    in
    let background =
      Array.init (1 + Random.State.int rand 5) ~f:(fun _ -> random_row ~n_features rand)
    in
    let bg = Tree_shap.Background.create forest background |> Or_error.ok_exn in
    for _ = 1 to 5 do
      let x = random_row ~n_features rand in
      let into = Array.create ~len:n_features 0. in
      let margin = Tree_shap.interventional bg x ~into in
      let expected = shapley ~n:n_features (interventional_game forest ~background x) in
      worst := Float.max !worst (worst_diff into expected);
      let sum = Tree_shap.Background.base bg +. Array.sum (module Float) into ~f:Fn.id in
      worst_sum := Float.max !worst_sum (Float.abs (sum -. margin))
    done
  done;
  printf "max |phi - shapley| < 1e-12: %b\n" Float.(!worst < 1e-12);
  printf "max |base + sum phi - margin| < 1e-12: %b\n" Float.(!worst_sum < 1e-12);
  [%expect
    {|
    max |phi - shapley| < 1e-12: true
    max |base + sum phi - margin| < 1e-12: true
    |}]
;;

let%expect_test "a forest without covers has no path-dependent workspace" =
  let forest =
    Forest.create
      ~n_features:1
      ~base_score:0.
      ~roots:[| 0 |]
      ~feature:[| 0; -1; -1 |]
      ~threshold:[| 0.; 0.; 0. |]
      ~default_left:[| true; true; true |]
      ~left:[| 1; -1; -1 |]
      ~right:[| 2; -1; -1 |]
      ~value:[| 0.5; 0.; 1. |]
      ()
    |> Or_error.ok_exn
  in
  print_s
    [%sexp
      (Tree_shap.workspace forest |> Or_error.map ~f:(fun _ -> ()) : unit Or_error.t)];
  [%expect
    {|
    (Error
     "the model has no node covers; re-export it to get path-dependent attributions")
    |}]
;;

let%expect_test "path-dependent attributions allocate nothing" =
  let rand = Random.State.make [| 3 |] in
  let forest = random_forest ~n_features:6 ~trees:20 ~max_depth:6 rand in
  let ws = Tree_shap.workspace forest |> Or_error.ok_exn in
  let x = random_row ~n_features:6 rand in
  let into = Array.create ~len:6 0. in
  let (_ : float#) = Tree_shap.path_dependent_u ws x ~into in
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    let (_ : float#) = Tree_shap.path_dependent_u ws x ~into in
    ()
  done;
  printf "words: %d\n" (Gc.minor_words () - before);
  [%expect {| words: 0 |}]
;;
