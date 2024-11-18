import torch
import torch.nn as nn
from torch.nested import nested_tensor


def edgeify(*arg_is_nested_tensor):
  first_nested_arg = arg_is_nested_tensor.index(True)
  def edgify_fn(fn):
    def apply_fn(*args):
      return nested_tensor([
        fn(*[
          (arg[i] if is_nested else arg)
          for arg, is_nested in zip(args, arg_is_nested_tensor)])
        for i in range(args[first_nested_arg].size(0))
      ])
    return apply_fn
  return edgify_fn


@edgeify(True, False)
def _edgefn_radial_encode_edge(x, npi):
  x_sq = (x**2).sum(-1)
  return torch.cos(npi*torch.sqrt(x_sq)[..., None])*torch.relu(1. - x_sq)[..., None]
def radial_encode_edge(r, n, rmax):
  """ r: (..., 3)
      ans: (..., n)"""
  npi = torch.pi*torch.arange(0, n, device=r.device)
  x = r/rmax
  return _edgefn_radial_encode_edge(x, npi)


@edgeify(False, False, True)
def _edgefn_einsum_weight_nested(einsum_str, weight, x_nested):
  return torch.einsum(einsum_str, weight, x_nested)
class EdgeTensLinear(nn.Module):
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
    return _edgefn_einsum_weight_nested(self.einsum_str, self.W, x)

@edgeify(False, True, True)
def _edgefn_einsum_nested_nested(einsum_str, x_l_nested, x_r_nested):
  return torch.einsum(einsum_str, x_l_nested, x_r_nested)
class EdgeTensorProds(nn.Module):
  def __init__(self, inds_l:int, inds_r:int, inds_o:int, dim_l:int, dim_r:int, dim_o:int, rank:int):
    super().__init__()
    def assert_div_2(a):
      assert a%2 == 0, "parity mismatch"
      return a//2
    contracts = assert_div_2(inds_l + inds_r - inds_o) # number of tensor contractions
    assert 0 <= contracts <= min(inds_l, inds_r), f"{contracts} contractions is out of range for {inds_l}, {inds_r} index tensors"
    self.lin_l = EdgeTensLinear(inds_l, dim_l, rank)
    self.lin_r = EdgeTensLinear(inds_r, dim_r, rank)
    self.lin_o = EdgeTensLinear(inds_o, rank, dim_o)
    tensor_indices = "".join([chr(ord("i") + n) for n in range(inds_l + inds_r - contracts)])
    to_contract = tensor_indices[:contracts]
    keep_l      = tensor_indices[contracts:inds_l]
    keep_r      = tensor_indices[inds_l:]
    self.einsum_str = f"...{keep_l + to_contract}, ...{keep_r + to_contract} -> ...{keep_l + keep_r}"
  def forward(self, x_l, x_r):
    x_l = self.lin_l(x_l)
    x_r = self.lin_r(x_r)
    x_o = _edgefn_einsum_nested_nested(self.einsum_str, x_l, x_r)
    return self.lin_o(x_o)
