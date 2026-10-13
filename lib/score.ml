open! Core

module Stable = struct
  module Reason = struct
    module V1 = struct
      type t =
        { feature : string
        ; contribution : float
        }
      [@@deriving bin_io, sexp, compare, equal]
    end
  end

  module V1 = struct
    type t =
      { model : string
      ; score : float
      ; margin : float
      ; reasons : Reason.V1.t list
      }
    [@@deriving bin_io, sexp, compare, equal]
  end
end

module Reason = Stable.Reason.V1
include Stable.V1

let top_reasons ~names contributions ~k =
  Array.filter_mapi contributions ~f:(fun i c -> Option.some_if Float.(c > 0.) (c, i))
  |> Array.to_list
  |> List.sort ~compare:(fun (c1, i1) (c2, i2) ->
    match Float.compare c2 c1 with
    | 0 -> Int.compare i1 i2
    | n -> n)
  |> (fun l -> List.take l k)
  |> List.map ~f:(fun (contribution, i) -> { Reason.feature = names.(i); contribution })
;;
