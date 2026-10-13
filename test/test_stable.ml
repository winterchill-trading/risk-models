open! Core
open Risk_models

(* Pinned bin_io shape digests: a change here means journaled scores or calibration tables
   written by older code no longer read. Add a V2 instead of editing V1. *)
let%expect_test "stable shapes" =
  let show name shape =
    printf "%s %s\n" name (Bin_prot.Shape.eval_to_digest_string shape)
  in
  show "Score.V1" Score.Stable.V1.bin_shape_t;
  show "Score.Reason.V1" Score.Stable.Reason.V1.bin_shape_t;
  show "Schema.V1" Schema.Stable.V1.bin_shape_t;
  show "Calibration.V1" Calibration.Stable.V1.bin_shape_t;
  [%expect
    {|
    Score.V1 7ec5dc1f0acb53a323e8862cf6be25e3
    Score.Reason.V1 85c426d787cdbd7c8626b645e113df79
    Schema.V1 5237808a62a7323de50188526f240c9c
    Calibration.V1 2fe07b0e56f348624fed318804cd88ae
    |}]
;;

let%expect_test "a score round-trips through bin_io" =
  let s =
    { Score.model = "m"
    ; score = 0.25
    ; margin = -1.5
    ; reasons = [ { feature = "amount"; contribution = 0.75 } ]
    }
  in
  let back =
    Binable.of_string
      (module Score.Stable.V1)
      (Binable.to_string (module Score.Stable.V1) s)
  in
  print_s [%sexp (Score.equal s back : bool)];
  [%expect {| true |}]
;;
