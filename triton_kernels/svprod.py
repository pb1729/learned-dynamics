import torch
import triton
import triton.language as tl
from typing import Tuple


# TODO: make reference implementation, then finish the rest of this...


class _svprod(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x_a:torch.Tensor, x_v:torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Triton implementation of all combinations of the scalar vector product
        l_a, r_a: (batch, nodes, chan)
        l_v, r_v: (batch, nodes, chan, 3)
        w: (out_chan, chan, 5) -- [aa, vv; av, va, vv]
        return ya, y_v
        y_a: (batch, nodes, chan)
        y_v: (batch, nodes, chan, 3) """
    return super().forward(*args, **kwargs)
