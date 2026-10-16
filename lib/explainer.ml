open! Core
module Float_u = Stdlib_upstream_compatible.Float_u

type body =
  | Path of Tree_shap.workspace
  | Background of Tree_shap.Background.t
  | Linear of Linear.t

type t =
  { name : string
  ; version : string
  ; names : string array
  ; body : body
  ; base : float
  ; scorer : Scorer.t option
  }

let tree ?background forest =
  match background with
  | None ->
    Tree_shap.workspace forest |> Or_error.map ~f:(fun ws -> Path ws, Forest.bias forest)
  | Some rows ->
    Tree_shap.Background.create forest rows
    |> Or_error.map ~f:(fun bg -> Background bg, Tree_shap.Background.base bg)
;;

let of_forest ?background ~name ~version ~names forest =
  let open Or_error.Let_syntax in
  let%bind () =
    if Array.length names = Forest.n_features forest
    then Ok ()
    else
      Or_error.error_s
        [%message
          "feature names do not match the forest"
            ~names:(Array.length names : int)
            ~features:(Forest.n_features forest : int)]
  in
  let%map body, base = tree ?background forest in
  { name; version; names; body; base; scorer = None }
;;

let create ?background scorer =
  let open Or_error.Let_syntax in
  let%map body, base =
    match Scorer.body scorer with
    | Forest f -> tree ?background f
    | Linear l -> Ok (Linear l, Linear.intercept l)
  in
  { name = Scorer.name scorer
  ; version = Scorer.version scorer
  ; names = (Scorer.schema scorer).features
  ; body
  ; base
  ; scorer = Some scorer
  }
;;

let method_ t : Explanation.Method.t =
  match t.body with
  | Path _ -> Path_dependent
  | Background _ -> Interventional
  | Linear _ -> Linear
;;

let base t = t.base

let[@zero_alloc] contributions_u t x ~into =
  match t.body with
  | Path ws -> Tree_shap.path_dependent_u ws x ~into
  | Background bg -> Tree_shap.interventional_u bg x ~into
  | Linear l -> Linear.contributions_u l x ~into
;;

let explain ?k ?label t x =
  let into = Array.create ~len:(Array.length t.names) 0. in
  let margin = Float_u.to_float (contributions_u t x ~into) in
  let score =
    Option.map t.scorer ~f:(fun s ->
      Calibration.apply (Scorer.calibration s) (Calibration.sigmoid margin))
  in
  Explanation.create
    ?k
    ?label
    ?score
    ~model:t.name
    ~version:t.version
    ~method_:(method_ t)
    ~names:t.names
    ~x
    ~base:t.base
    ~margin
    into
;;
