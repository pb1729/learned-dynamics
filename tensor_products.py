from itertools import permutations

import torch
import torch.nn as nn

from layers_common import VecLinear


# define the Levi-Civita symbol
ε_ijk = torch.zeros(3, 3, 3, device="cuda")
for i, j, k in permutations(range(3)):
  ε_ijk[i, j, k] = ((i-j)/abs(i-j))*((j-k)/abs(j-k))*((k-i)/abs(k-i))

def ε_d2v(x_d):
  return torch.einsum("...ij, ijk -> ...k", x_d, ε_ijk)

def ε_v2d(x_v):
  return torch.einsum("...i, ijk -> ...kj", x_v, ε_ijk)

class ChiralMix(nn.Module):
  """ Acts on inputs x_v with 1 index and x_d with 2 indices.
      Introduce chirality by contracting with Levi-Civita symbol.
      For completeness, we also throw in a linear transform on x_d^T. """
  def __init__(self, dim_v, dim_d):
    super().__init__()
    self.lin_vd = VecLinear(dim_v, dim_d)
    self.lin_dv = VecLinear(dim_d, dim_v)
    self.lin_dd = TensLinear(2, dim_d, dim_d)
  def forward(self, x_v, x_d):
    return self.lin_dv(ε_d2v(x_d)), ε_v2d(self.lin_vd(x_v)) + torch.transpose(self.lin_dd(x_d), -1, -2)


class TensLinear(nn.Module):
  def __init__(self, inds, dim_in, dim_out):
    super().__init__()
    self.W = nn.Parameter(torch.empty(dim_out, dim_in))
    tensor_indices = "".join([chr(ord("i") + n) for n in range(inds)]) # eg "ijk" if inds==3
    self.einsum_str = f"OI, ...I{tensor_indices} -> ...O{tensor_indices}"
    self.self_init()
  def self_init(self):
    _, fan_in = self.W.shape
    nn.init.normal_(self.W, std=fan_in**-0.5)
  def forward(self, x):
    return torch.einsum(self.einsum_str, self.W, x)


class TensorProds(nn.Module):
  def __init__(self, inds_l:int, inds_r:int, inds_o:int, dim_l:int, dim_r:int, dim_o:int, rank:int):
    super().__init__()
    def assert_div_2(a):
      assert a%2 == 0, "parity mismatch"
      return a//2
    contracts = assert_div_2(inds_l + inds_r - inds_o) # number of tensor contractions
    assert 0 <= contracts <= min(inds_l, inds_r), f"{contracts} contractions is out of range for {inds_l}, {inds_r} index tensors"
    self.lin_l = TensLinear(inds_l, dim_l, rank)
    self.lin_r = TensLinear(inds_r, dim_r, rank)
    self.lin_o = TensLinear(inds_o, rank, dim_o)
    tensor_indices = "".join([chr(ord("i") + n) for n in range(inds_l + inds_r - contracts)])
    to_contract = tensor_indices[:contracts]
    keep_l      = tensor_indices[contracts:inds_l]
    keep_r      = tensor_indices[inds_l:]
    self.einsum_str = f"...{keep_l + to_contract}, ...{keep_r + to_contract} -> ...{keep_l + keep_r}"
  def forward(self, x_l, x_r):
    x_l = self.lin_l(x_l)
    x_r = self.lin_r(x_r)
    x_o = torch.einsum(self.einsum_str, x_l, x_r)
    # TODO: if this is causing instability, we *could* divide by sqrt(1 + x_l^2 + x_r^2)
    return self.lin_o(x_o)


if __name__ == "__main__":
  print(ε_ijk)
