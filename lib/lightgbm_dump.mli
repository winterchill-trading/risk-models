(** Reads a LightGBM binary classifier from [Booster.dump_model()] JSON into a {!Forest}.

    Accepted: objective ["binary sigmoid:1"], one tree per iteration, no averaging (so not
    random-forest mode), numerical splits with [decision_type "<="] only. Anything else
    (multiclass, categorical splits, other objectives) is refused rather than scored
    wrongly. The margin is the plain sum of leaf values (LightGBM folds the initial score
    into the first tree). Missing values follow each split's [missing_type]: ["NaN"],
    ["None"] (NaN read as 0) or ["Zero"]. Split-node values for reason codes are
    count-weighted subtree means. *)

open! Core

val of_yojson : n_features:int -> Yojson.Safe.t -> Forest.t Or_error.t
