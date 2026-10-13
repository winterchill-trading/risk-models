open! Core

module Repr = struct
  type t =
    { name : string
    ; version : int
    ; features : string array
    ; digest : string
    }
  [@@deriving bin_io, sexp, compare, equal]
end

type t = Repr.t =
  { name : string
  ; version : int
  ; features : string array
  ; digest : string
  }
[@@deriving sexp_of, compare, equal]

let digest_of_names names = Md5.digest_string (String.concat ~sep:"\n" names) |> Md5.to_hex

let create ~name ~version names =
  match names with
  | [] -> Or_error.error_s [%message "a feature schema needs at least one feature" name]
  | _ ->
    (match List.find_a_dup names ~compare:String.compare with
     | Some dup -> Or_error.error_s [%message "duplicate feature name" name dup]
     | None ->
       Ok { name; version; features = Array.of_list names; digest = digest_of_names names })
;;

let create_exn ~name ~version names = create ~name ~version names |> Or_error.ok_exn
let length t = Array.length t.features
let index t f = Array.findi t.features ~f:(fun _ g -> String.equal f g) |> Option.map ~f:fst

let index_exn t f =
  match index t f with
  | Some i -> i
  | None -> raise_s [%message "unknown feature" f ~schema:(t.name : string)]
;;

let check_compatible ~expected t =
  if String.equal expected.digest t.digest
  then Ok ()
  else
    Or_error.error_s
      [%message
        "feature schema mismatch"
          ~expected:(expected.name, expected.version, expected.digest : string * int * string)
          ~got:(t.name, t.version, t.digest : string * int * string)]
;;

let txn_features_v1 =
  create_exn
    ~name:"txn-features"
    ~version:1
    [ "amount_log10"
    ; "hour_sin"
    ; "hour_cos"
    ; "is_weekend"
    ; "is_card"
    ; "is_online"
    ; "is_crypto"
    ; "beneficiary_new"
    ; "beneficiary_age_days_log"
    ; "txn_count_1h"
    ; "txn_count_24h"
    ; "amount_sum_24h_log10"
    ; "amount_ratio_30d_log10"
    ; "distinct_beneficiaries_24h"
    ; "inbound_24h_log10"
    ; "inbound_share_of_balance"
    ; "near_threshold"
    ; "near_threshold_count_7d"
    ; "new_device"
    ; "hours_since_credential_change_log"
    ; "ip_country_mismatch"
    ; "foreign_merchant"
    ; "mcc_high_risk"
    ; "account_age_days_log"
    ; "balance_drain_ratio"
    ]
;;

module Stable = struct
  module V1 = struct
    type t = Repr.t [@@deriving bin_io, compare, equal, sexp_of]

    (* Re-derive the digest on read so a hand-edited sexp cannot carry a stale one. *)
    let t_of_sexp sexp =
      let r = Repr.t_of_sexp sexp in
      create_exn ~name:r.name ~version:r.version (Array.to_list r.features)
    ;;
  end
end
