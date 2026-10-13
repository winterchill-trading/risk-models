(** What a model says about one payment: the calibrated score, the raw margin and the
    features that pushed the score up the most. Bin_io so it can be journaled next to the
    rule decision it fed. *)

open! Core

module Reason : sig
  type t =
    { feature : string
    ; contribution : float (** log-odds added by this feature, > 0 *)
    }
  [@@deriving sexp, compare, equal]
end

type t =
  { model : string (** model name *)
  ; score : float (** calibrated probability *)
  ; margin : float (** log-odds before calibration *)
  ; reasons : Reason.t list (** largest contribution first, at most three by default *)
  }
[@@deriving sexp, compare, equal]

(** The [k] largest positive contributions, largest first, ties by feature index. *)
val top_reasons : names:string array -> float array -> k:int -> Reason.t list

module Stable : sig
  module Reason : sig
    module V1 : sig
      type t = Reason.t [@@deriving bin_io, sexp, compare, equal]
    end
  end

  module V1 : sig
    type nonrec t = t [@@deriving bin_io, sexp, compare, equal]
  end
end
