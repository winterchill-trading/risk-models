open! Core
open Risk_models

(* Pinned bin_io shape digests: a change here means journaled scores or calibration tables
   written by older code no longer read. Add a V2 instead of editing V1. *)
let%expect_test "stable shapes" =
  let show name shape =
    printf "%s %s\n" name (Bin_prot.Shape.eval_to_digest_string shape)
  in
  show "Schema.V1" Schema.Stable.V1.bin_shape_t;
  show "Calibration.V1" Calibration.Stable.V1.bin_shape_t;
  show "Explanation.V1" Explanation.Stable.V1.bin_shape_t;
  [%expect
    {|
    Schema.V1 5237808a62a7323de50188526f240c9c
    Calibration.V1 2fe07b0e56f348624fed318804cd88ae
    Explanation.V1 cc0a0cb3e002565614a05fdaf17fdc6c
    |}]
;;
