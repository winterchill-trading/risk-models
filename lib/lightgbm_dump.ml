open! Core

type node =
  | Leaf of
      { value : float
      ; count : float
      }
  | Split of
      { feature : int
      ; threshold : float
      ; default_left : bool
      ; missing : Forest.Missing.t
      ; count : float
      ; left : node
      ; right : node
      }

let count = function
  | Leaf { count; _ } | Split { count; _ } -> count
;;

let rec node_of_yojson json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let has k = Result.is_ok (member k json) in
  if has "leaf_value"
  then (
    let%bind value = field "leaf_value" float json in
    (* A tree with a single leaf has no count; it never splits, so the weight is moot. *)
    let%map count = if has "leaf_count" then field "leaf_count" float json else Ok 1. in
    Leaf { value; count })
  else (
    let%bind decision = field "decision_type" string json in
    let%bind () =
      if String.equal decision "<="
      then Ok ()
      else Or_error.error_s [%message "unsupported decision_type (categorical?)" decision]
    in
    let%bind feature = field "split_feature" int json in
    let%bind threshold = field "threshold" float json in
    let%bind default_left =
      field
        "default_left"
        (function
          | `Bool b -> Ok b
          | j ->
            Or_error.error_s [%message "expected a bool" ~got:(Yojson.Safe.to_string j)])
        json
    in
    let%bind missing =
      match%bind field "missing_type" string json with
      | "NaN" -> Ok Forest.Missing.Nan_default
      | "None" -> Ok Forest.Missing.Nan_as_zero
      | "Zero" -> Ok Forest.Missing.Zero_default
      | m -> Or_error.error_s [%message "unknown missing_type" m]
    in
    let%bind count = field "internal_count" float json in
    let%bind left = field "left_child" node_of_yojson json in
    let%map right = field "right_child" node_of_yojson json in
    Split { feature; threshold; default_left; missing; count; left; right })
;;

(* Count-weighted mean of the leaves below; the same formula as the Python reference
   scorer, so reason codes match to the bit. *)
let rec expected = function
  | Leaf { value; _ } -> value
  | Split { left; right; _ } ->
    let cl = count left
    and cr = count right in
    ((cl *. expected left) +. (cr *. expected right)) /. (cl +. cr)
;;

let of_yojson ~n_features json =
  let open Or_error.Let_syntax in
  let open Json_util in
  let%bind objective = field "objective" string json in
  let%bind () =
    if String.equal objective "binary sigmoid:1"
    then Ok ()
    else Or_error.error_s [%message "unsupported LightGBM objective" objective]
  in
  let%bind per_iteration = field "num_tree_per_iteration" int json in
  let%bind average =
    field
      "average_output"
      (function
        | `Bool b -> Ok b
        | _ -> Ok true)
      json
  in
  let%bind () =
    if per_iteration = 1 && not average
    then Ok ()
    else
      Or_error.error_s
        [%message "only single-output, non-averaged boosters are supported"]
  in
  let%bind trees =
    field
      "tree_info"
      (list (fun t ->
         let%bind num_cat = field "num_cat" int t in
         if num_cat > 0
         then Or_error.error_s [%message "categorical splits are not supported"]
         else field "tree_structure" node_of_yojson t))
      json
  in
  (* Flatten in preorder: every child gets a larger index than its parent. *)
  let feature = Queue.create ()
  and threshold = Queue.create ()
  and default_left = Queue.create ()
  and missing = Queue.create ()
  and left = Queue.create ()
  and right = Queue.create ()
  and value = Queue.create ()
  and cover = Queue.create () in
  let rec emit node =
    let i = Queue.length feature in
    Queue.enqueue value (expected node);
    Queue.enqueue cover (count node);
    match node with
    | Leaf _ ->
      Queue.enqueue feature (-1);
      Queue.enqueue threshold 0.;
      Queue.enqueue default_left false;
      Queue.enqueue missing Forest.Missing.Nan_default;
      Queue.enqueue left (-1);
      Queue.enqueue right (-1);
      i
    | Split s ->
      Queue.enqueue feature s.feature;
      Queue.enqueue threshold s.threshold;
      Queue.enqueue default_left s.default_left;
      Queue.enqueue missing s.missing;
      Queue.enqueue left (-1);
      Queue.enqueue right (-1);
      let l = emit s.left in
      let r = emit s.right in
      Queue.set left i l;
      Queue.set right i r;
      i
  in
  let roots = List.map trees ~f:emit |> Array.of_list in
  Forest.create
    ~n_features
    ~base_score:0.
    ~roots
    ~feature:(Queue.to_array feature)
    ~threshold:(Queue.to_array threshold)
    ~default_left:(Queue.to_array default_left)
    ~missing:(Queue.to_array missing)
    ~left:(Queue.to_array left)
    ~right:(Queue.to_array right)
    ~value:(Queue.to_array value)
    ~cover:(Queue.to_array cover)
    ()
;;
